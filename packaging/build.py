#!/usr/bin/env python3
"""Build UncleFu.app via venvstacks.

See `docs/decisions.md` for why we ship a layered Python runtime
(MLX namespace packages + ad-hoc dynamic linking require it).

Stages:
  1. Lock + build the three venvstacks layers
     (cpython-3.12 runtime, framework, application).
  2. Install mlx-audio --no-deps after the layer build
     (its mlx-lm pin conflicts with mlx-vlm's; resolver gives up).
  3. Create UncleFu.app/ bundle structure under dist/.
  4. Compile a Mach-O C launcher (CFBundleExecutable) that dlopens
     libpython and runs `-m unclefu` in-process. A bash launcher
     does NOT work for menubar apps — macOS LaunchServices won't grant
     WindowServer access to a shell script.
  5. Copy our source into Contents/Resources/unclefu/ (with sprites).
  6. Ad-hoc sign (`codesign -s -`) so TCC persists permission grants.

Usage:
    uv run python packaging/build.py [--skip-venv] [--clean]

The first build downloads + locks ~2 GB of wheels (~5-15 min). Subsequent
builds with --skip-venv reuse the export and just rebuild the bundle.

Real Developer ID signing + notarization + DMG packaging are on the
roadmap (so TCC grants persist across rebuilds and Gatekeeper accepts
the build without a right-click bypass).
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BUILD_DIR = SCRIPT_DIR / "_build"
EXPORT_DIR = SCRIPT_DIR / "_export"
DIST_DIR = REPO_ROOT / "dist"
WHEELS_DIR = SCRIPT_DIR / "_wheels"
TOML_PATH = SCRIPT_DIR / "venvstacks.toml"

APP_NAME = "UncleFu"            # display name + .app dir + launcher binary
APP_BUNDLE = f"{APP_NAME}.app"
APP_BUNDLE_ID = "dev.unclefu.app"
APP_VERSION = "0.1.0"
# Python package + SQLite path (~/Library/Application Support/UncleFu/)
# are keyed off the lowercase identifier "unclefu", separate from APP_NAME.
PYTHON_VERSION = "3.12"          # must match venvstacks.toml runtime
PYTHON_TAG = f"cpython-{PYTHON_VERSION}"
FRAMEWORK_LAYER = "unclefu-fw"
APP_LAYER = "unclefu-app"

# mlx-audio git pin. We build it as a wheel from this commit, then
# install --no-deps into the framework layer. Required because mlx-audio
# pins mlx-lm==0.31.1 which conflicts with mlx-vlm's own mlx-lm pin and
# the venvstacks resolver gives up.
MLX_AUDIO_GIT = (
    "git+https://github.com/Blaizzy/mlx-audio"
    "@51753266e0a4f766fd5e6fbc46652224efc23981"
)

# Sdist-only deps that need a local wheel build because venvstacks runs
# uv with `--only-binary :all:`. Add to this list if a future lock errors
# with "No solution found ... has no usable wheels and ... building from
# source is disabled".
SDIST_ONLY_DEPS = [
    "rumps==0.4.0",  # rumps publishes sdist only; 0.3.0 is the last wheel.
]


# ─── shell helpers ─────────────────────────────────────────────────────


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> None:
    """Run a command, streaming output. Bail on failure unless check=False."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if check and result.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}")


def have_cmd(name: str) -> bool:
    return shutil.which(name) is not None


# ─── clean ────────────────────────────────────────────────────────────


def clean_all(*, preserve_venv: bool = False) -> None:
    print("\n[clean] removing build artifacts…")
    dirs = [DIST_DIR]
    if not preserve_venv:
        dirs.extend([BUILD_DIR, EXPORT_DIR, WHEELS_DIR])
    for d in dirs:
        if d.exists():
            print(f"  rm -rf {d}")
            shutil.rmtree(d, ignore_errors=True)


# ─── stage 1: venvstacks layers ───────────────────────────────────────


def build_venvstacks() -> None:
    """Lock + build + export the three layers via venvstacks."""
    print("\n[1/4] building venvstacks layers …")

    if not have_cmd("pipx"):
        sys.exit(
            "pipx not found. Install with: brew install pipx && pipx ensurepath"
        )

    # Pre-build any sdist-only deps as local wheels — venvstacks runs uv
    # with `--only-binary :all:` so sdists from PyPI are rejected outright.
    _build_sdist_wheels()

    local_wheels_args: list[str] = []
    if WHEELS_DIR.exists() and any(WHEELS_DIR.glob("*.whl")):
        local_wheels_args = ["--local-wheels", str(WHEELS_DIR)]

    # Lock — if-needed so we re-use a previous lock when possible.
    print("\n  locking …")
    run([
        "pipx", "run", "venvstacks", "lock",
        str(TOML_PATH), "--if-needed",
    ] + local_wheels_args)

    # Build the layered environments.
    print("\n  building (slow; first run downloads wheels) …")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    run([
        "pipx", "run", "venvstacks", "build",
        str(TOML_PATH), "--no-lock",
        "--build-dir", str(BUILD_DIR),
    ] + local_wheels_args)

    # Export the layers as plain directories for bundling.
    print("\n  exporting …")
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    run([
        "pipx", "run", "venvstacks", "local-export",
        str(TOML_PATH),
        "--build-dir", str(BUILD_DIR),
        "--output-dir", str(EXPORT_DIR),
    ])

    # mlx-audio --no-deps because its mlx-lm pin conflicts.
    _install_mlx_audio()


def _build_sdist_wheels() -> None:
    """Build wheels from sdist-only PyPI deps so venvstacks (which runs
    uv with --only-binary :all:) can resolve them."""
    if not SDIST_ONLY_DEPS:
        return
    WHEELS_DIR.mkdir(exist_ok=True)
    if not have_cmd("uv"):
        sys.exit("uv not found on PATH; can't build sdist wheels.")
    for spec in SDIST_ONLY_DEPS:
        # Skip if a wheel already exists for this spec.
        pkg = spec.split("==", 1)[0].replace("-", "_")
        if list(WHEELS_DIR.glob(f"{pkg}-*.whl")):
            print(f"  cached wheel for {spec}")
            continue
        print(f"  building wheel for {spec} …")
        # `uv pip` doesn't ship a `wheel` subcommand and our uv-managed
        # venv has no pip. uvx runs pip from an isolated ephemeral env.
        run([
            "uvx", "--from", "pip", "pip", "wheel",
            "--no-deps", "--wheel-dir", str(WHEELS_DIR), spec,
        ])


def _install_mlx_audio() -> None:
    """Build mlx-audio from git, install into the exported framework layer
    with --no-deps. Same dance oMLX does."""
    print("\n  installing mlx-audio --no-deps …")
    fw_dir = EXPORT_DIR / f"framework-{FRAMEWORK_LAYER}"
    site_packages = fw_dir / "lib" / f"python{PYTHON_VERSION}" / "site-packages"
    if not site_packages.exists():
        sys.exit(f"framework site-packages missing: {site_packages}")

    WHEELS_DIR.mkdir(exist_ok=True)
    # Skip the build if we already have a wheel from a previous run.
    wheels = sorted(WHEELS_DIR.glob("mlx_audio-*.whl"))
    if not wheels:
        run([
            "uvx", "--from", "pip", "pip", "wheel", "--no-deps",
            "--wheel-dir", str(WHEELS_DIR), MLX_AUDIO_GIT,
        ])
        wheels = sorted(WHEELS_DIR.glob("mlx_audio-*.whl"))
    if not wheels:
        sys.exit("mlx-audio wheel build produced no .whl")
    whl = wheels[-1]
    print(f"  installing {whl.name} into {site_packages}")
    run([
        "uvx", "--from", "pip", "pip", "install",
        "--no-deps", "--target", str(site_packages),
        "--upgrade", str(whl),
    ])


# ─── stage 2: bundle ──────────────────────────────────────────────────


def create_app_bundle() -> Path:
    print("\n[2/4] assembling app bundle …")
    app_dir = DIST_DIR / APP_BUNDLE
    contents = app_dir / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    lib = contents / "lib"

    # NB: venvstacks layers live under Resources/, NOT Frameworks/.
    # codesign treats every directory under Frameworks/ as a signable
    # bundle (.framework / .dylib / .app); venvstacks layers are
    # regular Python virtualenvs full of JSON metadata, headers, and
    # py source, which fails that validation. Resources/ has no such
    # constraint and the layers work identically from there because
    # all internal paths are @executable_path-relative.
    if app_dir.exists():
        shutil.rmtree(app_dir)
    for d in (macos, resources, lib):
        d.mkdir(parents=True)

    # Copy the three layer dirs into Resources/.
    print("  copying layer environments …")
    for layer in (PYTHON_TAG,
                  f"framework-{FRAMEWORK_LAYER}",
                  f"app-{APP_LAYER}"):
        src = EXPORT_DIR / layer
        if not src.exists():
            sys.exit(f"missing exported layer: {src}")
        dst = resources / layer
        shutil.copytree(src, dst, symlinks=True)
        print(f"    {layer}")

    # Venvstacks export-metadata dir. Same Resources/ reasoning.
    meta = EXPORT_DIR / "__venvstacks__"
    if meta.exists():
        shutil.copytree(meta, resources / "__venvstacks__", symlinks=True)

    # Copy our source into Resources/unclefu. Sprites included via the
    # MANIFEST that python's recursive copy picks up.
    print("  copying unclefu source …")
    src_pkg = REPO_ROOT / "src" / "unclefu"
    dst_pkg = resources / "unclefu"
    shutil.copytree(
        src_pkg, dst_pkg,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", ".DS_Store",
        ),
    )

    # Copy the bundled python3 into MacOS/ so launchd sees it as part of
    # the bundle. The C launcher will dlopen libpython directly; this is
    # for any subprocess that wants `python3` (and so macOS doesn't get
    # confused about what's executable in the bundle).
    src_python = resources / PYTHON_TAG / "bin" / "python3"
    dst_python = macos / "python3"
    if src_python.exists():
        shutil.copy2(src_python, dst_python)
        dst_python.chmod(0o755)

    # The bundled python3 references @executable_path/../lib/libpython3.12.dylib.
    # Symlink it from Contents/lib/ → Resources/cpython-3.12/lib/.
    real_libpy = resources / PYTHON_TAG / "lib" / f"libpython{PYTHON_VERSION}.dylib"
    if real_libpy.exists():
        (lib / f"libpython{PYTHON_VERSION}.dylib").symlink_to(
            f"../Resources/{PYTHON_TAG}/lib/libpython{PYTHON_VERSION}.dylib"
        )

    _compile_launcher(macos)
    has_icon = _install_icon(resources)
    _write_info_plist(contents, has_icon=has_icon)
    _adhoc_sign(app_dir)
    print(f"  ✓ {app_dir}")
    return app_dir


def _adhoc_sign(app_dir: Path) -> None:
    """Apply an ad-hoc signature to the bundle.

    macOS Sequoia's TCC may silently drop permission entries for
    unsigned bundles — meaning the Camera / Screen Recording prompts
    appear, the user grants, and the next launch acts as if nothing
    happened. Ad-hoc signing (`codesign -s -`) is the minimum bar
    that gets the bundle id registered in TCC.

    Caveat for dev: every rebuild produces a fresh ad-hoc signature,
    which macOS treats as a different application. The user has to
    re-grant permissions after each rebuild. The only way to keep
    grants stable across rebuilds is real Developer ID + notarisation.
    """
    print("  ad-hoc signing bundle …")
    # We deliberately DO NOT enable --options=runtime: the Hardened
    # Runtime blocks AVFoundation camera + ScreenCaptureKit APIs unless
    # we ship com.apple.security.device.camera and ...screen-capture
    # entitlements (which only count with a real Developer ID). For
    # ad-hoc dev, plain signing is what we want.
    #
    # We also AVOID --deep. The Frameworks/ tree contains a Python
    # include dir (include/python3.12/) that --deep mistakes for a
    # malformed framework and aborts on. TCC only needs the main
    # bundle's signature for permission entries to persist, so we
    # sign the top-level .app and the launcher executable explicitly.
    launcher = app_dir / "Contents" / "MacOS" / APP_NAME
    if launcher.exists():
        run(["codesign", "--sign", "-", "--force", str(launcher)])
    run(["codesign", "--sign", "-", "--force", str(app_dir)])
    # Quick sanity print — not gating the build.
    run(["codesign", "-dv", str(app_dir)], check=False)


def _compile_launcher(macos_dir: Path) -> None:
    """Compile a Mach-O launcher that dlopens libpython and runs
    `-m unclefu`. Lifted from oMLX with strings updated. A bash
    launcher does NOT work for menubar apps."""
    print("  compiling C launcher …")
    launcher_c = macos_dir / "_launcher.c"
    launcher_c.write_text(f'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>

typedef int (*py_bytes_main_fn)(int, char **);

static void show_error(const char *msg) {{
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "osascript -e 'display dialog \\"%s\\" buttons {{\\"OK\\"}} "
        "default button 1 with icon stop with title \\"{APP_NAME}\\"'",
        msg);
    system(cmd);
}}

int main(int argc, char *argv[]) {{
    char exe_buf[PATH_MAX];
    char resolved[PATH_MAX];
    uint32_t size = sizeof(exe_buf);

    if (_NSGetExecutablePath(exe_buf, &size) != 0) {{
        show_error("Failed to get executable path."); return 1;
    }}
    if (!realpath(exe_buf, resolved)) {{
        show_error("Failed to resolve executable path."); return 1;
    }}

    /* Trim executable name to get MacOS/ dir */
    char *slash = strrchr(resolved, '/');
    if (!slash) {{ show_error("Invalid path."); return 1; }}
    *slash = '\\0';
    char macos_dir[PATH_MAX];
    strncpy(macos_dir, resolved, sizeof(macos_dir) - 1);

    /* Trim MacOS to get Contents/ dir */
    slash = strrchr(resolved, '/');
    if (!slash) {{ show_error("Invalid bundle structure."); return 1; }}
    *slash = '\\0';
    char contents_dir[PATH_MAX];
    strncpy(contents_dir, resolved, sizeof(contents_dir) - 1);

    /* Layers live under Contents/Resources/ (NOT Frameworks/ — codesign
       treats anything in Frameworks/ as a signable bundle and chokes on
       the JSON metadata + headers inside venvstacks layers). */
    char layers_dir[PATH_MAX];
    snprintf(layers_dir, sizeof(layers_dir), "%s/Resources", contents_dir);
    if (access(layers_dir, F_OK) != 0) {{
        show_error("Python runtime not found in app bundle."); return 1;
    }}

    char pythonhome[PATH_MAX];
    snprintf(pythonhome, sizeof(pythonhome),
             "%s/{PYTHON_TAG}", layers_dir);
    setenv("PYTHONHOME", pythonhome, 1);

    char pythonpath[PATH_MAX * 4];
    snprintf(pythonpath, sizeof(pythonpath),
        "%s/Resources:"
        "%s/app-{APP_LAYER}/lib/python{PYTHON_VERSION}/site-packages:"
        "%s/framework-{FRAMEWORK_LAYER}/lib/python{PYTHON_VERSION}/site-packages",
        contents_dir, layers_dir, layers_dir);
    setenv("PYTHONPATH", pythonpath, 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);

    /* Load bundled libpython, run -m unclefu in-process. */
    char libpython[PATH_MAX];
    snprintf(libpython, sizeof(libpython),
             "%s/lib/libpython{PYTHON_VERSION}.dylib", contents_dir);
    void *py = dlopen(libpython, RTLD_NOW | RTLD_GLOBAL);
    if (!py) {{
        char err[1024];
        snprintf(err, sizeof(err), "Failed to load libpython: %s", dlerror());
        show_error(err);
        return 1;
    }}
    py_bytes_main_fn py_bytes_main = (py_bytes_main_fn)dlsym(py, "Py_BytesMain");
    if (!py_bytes_main) {{
        char err[1024];
        snprintf(err, sizeof(err),
                 "Failed to resolve Py_BytesMain: %s", dlerror());
        show_error(err);
        return 1;
    }}

    /* Forward terminal argv[1..] through to Python so flags like
       --debug / --skip-preflight work when launched from terminal.
       When double-clicked from /Applications, argc==1 and we just
       pass our default 3 args. */
    int base = 3;
    int new_argc = base + (argc - 1);
    char **py_argv = (char **)malloc(sizeof(char *) * (new_argc + 1));
    if (!py_argv) {{ show_error("Out of memory"); return 1; }}
    py_argv[0] = "{APP_NAME}";
    py_argv[1] = "-m";
    py_argv[2] = "unclefu";
    for (int i = 1; i < argc; i++) py_argv[base + i - 1] = argv[i];
    py_argv[new_argc] = NULL;
    return py_bytes_main(new_argc, py_argv);
}}
''')
    launcher_bin = macos_dir / APP_NAME
    run([
        "cc", "-arch", "arm64",
        "-mmacosx-version-min=15.0", "-O2",
        "-o", str(launcher_bin), str(launcher_c),
    ])
    launcher_c.unlink()
    launcher_bin.chmod(0o755)


def _install_icon(resources_dir: Path) -> bool:
    """Convert packaging/icon.png (1024×1024 RGBA) to AppIcon.icns and
    drop it in Resources/. Returns True if an icon was installed.

    macOS .icns is a multi-resolution container. We generate the ten
    canonical sizes (16/32/128/256/512 @1x and @2x), pack them via
    `iconutil -c icns`, then copy the result. iconutil is in /usr/bin
    on every modern macOS — no extra tooling needed."""
    src = SCRIPT_DIR / "icon.png"
    if not src.exists():
        print("  (no packaging/icon.png; skipping app icon)")
        return False
    try:
        from PIL import Image
    except ImportError:
        print("  (PIL not importable from build env; skipping icon)")
        return False

    print(f"  generating AppIcon.icns from {src.name} …")
    src_img = Image.open(src).convert("RGBA")
    if src_img.size != (1024, 1024):
        print(
            f"  ⚠ {src.name} is {src_img.size}, not 1024×1024. "
            "Resizing — icon may lose detail.",
        )
        src_img = src_img.resize((1024, 1024), Image.Resampling.LANCZOS)

    iconset_dir = BUILD_DIR / "AppIcon.iconset"
    if iconset_dir.exists():
        shutil.rmtree(iconset_dir)
    iconset_dir.mkdir(parents=True)

    # The Apple-canonical set. iconutil REQUIRES exactly these filenames.
    for size, suffix in (
        (16, "16x16"), (32, "16x16@2x"),
        (32, "32x32"), (64, "32x32@2x"),
        (128, "128x128"), (256, "128x128@2x"),
        (256, "256x256"), (512, "256x256@2x"),
        (512, "512x512"), (1024, "512x512@2x"),
    ):
        out = iconset_dir / f"icon_{suffix}.png"
        sized = src_img.resize((size, size), Image.Resampling.LANCZOS)
        sized.save(out, format="PNG", optimize=True)

    icns = resources_dir / "AppIcon.icns"
    run(["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns)])
    return True


def _write_info_plist(contents_dir: Path, *, has_icon: bool) -> None:
    """Bundle metadata. CFBundleIdentifier fixes rumps.notification too.

    NOTE: do NOT set LSUIElement — oMLX explicitly warns that combining
    it with runtime setActivationPolicy_ (which we do in __main__.py)
    breaks the NSStatusItem on Sonoma+. We rely on the runtime call.
    """
    plist = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": APP_BUNDLE_ID,
        "CFBundleVersion": APP_VERSION,
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "????",
        "LSMinimumSystemVersion": "15.0",
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
        "LSArchitecturePriority": ["arm64"],
        "NSCameraUsageDescription": (
            "Uncle Fu watches your posture and presence through the "
            "webcam to keep you on task. All processing is local."
        ),
        "NSScreenCaptureUsageDescription": (
            "Uncle Fu looks at your screens to notice when you drift "
            "from what you said you were focusing on. All processing is local."
        ),
        "NSHumanReadableCopyright": (
            f"Copyright © {datetime.now().year} Uncle Fu contributors."
        ),
    }
    if has_icon:
        plist["CFBundleIconFile"] = "AppIcon"
    with (contents_dir / "Info.plist").open("wb") as f:
        plistlib.dump(plist, f)


# ─── main ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-venv", action="store_true",
                        help="reuse the existing _export/ instead of rebuilding layers")
    parser.add_argument("--clean", action="store_true",
                        help="wipe _build/_export/_wheels/dist before building")
    args = parser.parse_args()

    if args.clean:
        clean_all()

    if not args.skip_venv:
        build_venvstacks()
    else:
        if not EXPORT_DIR.exists():
            sys.exit(
                f"--skip-venv but no _export at {EXPORT_DIR}. "
                f"Run without --skip-venv first."
            )

    app_dir = create_app_bundle()

    print(f"\n✓ built {app_dir}")
    print(f"\nNext:")
    print(f"  open {DIST_DIR}")
    print(f"  drag {APP_BUNDLE} to /Applications")
    print(f"  right-click → Open (Gatekeeper bypass, unsigned)")
    print(f"\nFirst launch downloads ~7 GB of model weights "
          f"into ~/.cache/huggingface.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
