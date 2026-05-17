# Sprite design plan

How we get from emoji-fallback to per-personality bespoke sprite art.

The taxonomy is locked (7 expressions per character — see
[../README.md](../README.md) and `vlm/schema.py`). What this doc describes
is the asset pipeline.

---

## Model choice: Nano Banana Pro (Gemini 3 Pro Image)

As of 2026-05, this is the character-consistency king: ~93% identity hold
across scenes, supports reference-image conditioning, exports grid images,
2K/4K output. The alternative — GPT-Image-2 — produces less consistent
sprite sheets and works better frame-by-frame.

### Quirks worth knowing before you fire prompts

- **No real alpha channel.** Both leading models lie about transparent
  backgrounds. The workaround that actually works: generate on a chroma
  green (`#00FF00`) background **with a thick white outline around the
  character**. White outline buffers the anti-aliasing halo when you key
  the green out.
- **Grids aren't pixel-uniform.** Don't divide a sheet by `width/4` — slice
  by detecting connected non-green regions.
- **JSON prompting is mostly Twitter folklore for these models.**
  Structured prose with labelled sections works better. The labels
  (CHARACTER / STYLE / LAYOUT / etc.) are what carries the structure, not
  curly braces.
- **Watermarks.** All Nano Banana outputs carry an invisible SynthID
  watermark — fine for commercial use, just means the image is detectable
  as AI-generated. Paid API tier removes the visible Gemini sparkle.
- **Silhouette legibility at 44 px is on the prompt-writer.** Specify it
  explicitly and verify by literally scaling the master down to 44 px and
  squinting before you commit to it.

---

## Workflow

Two prompts, one post-process. Iterate the master until it passes the
44-px silhouette test before moving on.

### Prompt 1 — master reference

Run with no input image. Pick best of N=4.

```
CHARACTER
Uncle Fu — a Chinese man in his early 60s. Lived-in face, salt-and-pepper hair
slightly thinning at the crown but fuller at the sides, neat. Small round
reading glasses sitting low on the nose, looking over them. Slight five
o'clock shadow, no full beard. Plain dark navy polo or buttoned shirt, top
button open. Warm but skeptical expression — eyebrows just slightly raised,
mouth set, faint half-smile that could turn into a sigh.

STYLE
Toy plastic / vinyl sticker style. Flat colors, no painterly gradients. Bold
clean shapes. Thick consistent white outline (3–4 px equivalent) around the
entire character silhouette. No background pattern, no shadows on the
character. Friendly proportions: head slightly larger than realistic, eyes
expressive.

FRAMING
Head and shoulders, centered, facing camera. Character fills ~80% of canvas
vertically. Aspect ratio 1:1.

BACKGROUND
Solid #00FF00 chroma green, no gradients, no noise, full bleed to edges.

OUTPUT REQUIREMENTS
- 2K resolution
- Distinct silhouette readable at 44 pixels: if you scaled this image to
  thumbnail size it must still be unambiguously Uncle Fu (glasses, hair
  shape, head shape).
- No text, no logos, no Gemini sparkle in the corner.
- Same character must be reproducible from this image as a reference.
```

Save the picked variant as `master.png`.

### Prompt 2 — expression sheet

Attach `master.png` as a reference image. Run once, regenerate individual
cells later if any drift.

```
CHARACTER
[reference image attached: master.png]
This is Uncle Fu. Use him exactly. Do not redesign the face, glasses, hair,
outfit, or proportions. Same vinyl sticker style, same white outline width,
same color palette as the reference.

LAYOUT
Produce one image containing 7 sprites of Uncle Fu, arranged in a 2-row grid:
- Row 1 (4 cells, left to right): IDLE, TALKING, DISAPPROVING, CONCERNED
- Row 2 (3 cells, left to right): SMIRK, APPROVING, ALARMED
Cells are clearly separated by green space. Each sprite is the same size,
head-and-shoulders, facing camera.

EXPRESSIONS
1. IDLE — neutral resting face. Looking ahead. Mouth closed, faint half-smile.
   This is the same face as the reference.
2. TALKING — mouth open mid-word, slightly animated. Eyes engaged, not
   scolding. Could be saying anything.
3. DISAPPROVING — frowning, eyebrows down. Looking *over* the glasses at the
   viewer. Mouth a flat line or slightly pursed. "Aiya, no" energy.
4. CONCERNED — softer face, eyebrows up and slightly together. Eyes worried
   but not angry. Mouth slightly open or pursed. "Are you ok?"
5. SMIRK — small knowing half-smile pulling up one side of the mouth. One
   eyebrow up. Eyes amused but not laughing. "I noticed."
6. APPROVING — small genuine smile, eyes warm. A reluctant little nod feel.
   "Good. Acceptable." Not beaming — Uncle Fu does not beam.
7. ALARMED — eyes wide, eyebrows shot up, mouth open in a small surprised "o".
   Looking directly at the viewer. "What are you doing??"

CONSISTENCY (CRITICAL)
- Same face shape, same glasses, same hair, same outfit, same color palette
  in all 7 cells.
- Identical white outline thickness on every sprite.
- Identical sprite size in every cell.
- Background everywhere is #00FF00 chroma green. No gradients, no decoration.

OUTPUT
- 4K resolution
- No labels, captions, numbers, or text overlays on the image.
- No Gemini sparkle.
```

Save as `sheet.png`.

---

## Post-process

The minimal path, no scripts:

1. **Chroma-key the whole sheet** with ImageMagick:
   ```
   magick sheet.png -fuzz 15% -transparent "#00FF00" sheet-keyed.png
   ```
   `-fuzz 15%` catches anti-alias halos around the green; tune up or down
   if edges look too tight or too smudgy.

2. **Crop each sprite manually** in Preview (or any image editor). Select
   each sprite with the rectangular selection, File → New from Clipboard,
   save as `<expression>.png` (lowercase, exact spelling from the taxonomy:
   `idle.png`, `talking.png`, `disapproving.png`, `concerned.png`,
   `smirk.png`, `approving.png`, `alarmed.png`).

3. **Resize each to 88 × 88** (44 pt @ 2x retina). ImageMagick:
   ```
   for f in *.png; do magick "$f" -resize 88x88 "$f"; done
   ```

4. **Drop into the repo:**
   ```
   src/unclefu/assets/personalities/uncle_fu/
   ```
   The menu bar picks them up next launch — no code change needed. The
   emoji fallback continues to apply for any missing file, so you can
   ship sprites one at a time if iteration drags.

5. **Silhouette QA at 44 px.** Open all 7 PNGs side by side in Preview at
   their final size. If any two are indistinguishable, regenerate the
   weaker one using the master as reference + only that expression's
   description block from Prompt 2.

---

## Future personalities

Same workflow per character. The `tools/bake_sprites.py` script (not yet
written — see `voice_design.md` for the matching `bake_voice.py` plan)
would automate steps 1–4 once the asset pipeline is worn-in enough to be
worth scripting. Until then, manual crop in Preview is fine — we only do
this once per character.
