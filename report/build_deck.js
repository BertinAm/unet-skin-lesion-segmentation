/* Build the project slide deck. Run: NODE_PATH=$(npm root -g) node report/build_deck.js */
const path = require("path");
const pptxgen = require("pptxgenjs");
const A = path.join(__dirname, "assets");

const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";

// Palette — "Teal Trust" (medical)
const NAVY = "0B2027", TEAL = "028090", SEA = "00A896", MINT = "02C39A",
      INK = "1A2E35", GRAY = "5A6B72", LIGHT = "F4F8F8", WHITE = "FFFFFF";
const HF = "Georgia", BF = "Calibri";
const W = 13.333, H = 7.5;

const slide = (bg = WHITE) => { const s = p.addSlide(); s.background = { color: bg }; return s; };
// kicker + title block
function head(s, kicker, title, dark = false) {
  s.addText(kicker.toUpperCase(), { x: 0.6, y: 0.45, w: 12, h: 0.35, fontFace: BF,
    fontSize: 13, bold: true, color: dark ? MINT : TEAL, charSpacing: 2 });
  s.addText(title, { x: 0.6, y: 0.8, w: 12.1, h: 0.9, fontFace: HF, fontSize: 30, bold: true,
    color: dark ? WHITE : INK });
}
function chip(s, x, y, w, h, fill, items) {
  s.addShape("roundRect", { x, y, w, h, fill: { color: fill }, line: { type: "none" },
    rectRadius: 0.08, shadow: { type: "outer", blur: 6, offset: 2, color: "AAAAAA", opacity: 0.3 } });
}

// ---- 1. TITLE -------------------------------------------------------------
let s = slide(NAVY);
s.addShape("rect", { x: 0, y: 0, w: 0.35, h: H, fill: { color: TEAL } });
s.addText("AUTOMATED MEDICAL IMAGE SEGMENTATION", { x: 0.9, y: 2.0, w: 11.5, h: 0.5,
  fontFace: BF, fontSize: 16, bold: true, color: MINT, charSpacing: 3 });
s.addText("Skin Lesion Segmentation with U-Net Variants", { x: 0.9, y: 2.55, w: 11.6, h: 1.2,
  fontFace: HF, fontSize: 44, bold: true, color: WHITE });
s.addText("U-Net  •  Attention U-Net  •  U-Net++     —     ISIC-2018 Dataset", { x: 0.9, y: 3.8,
  w: 11.6, h: 0.5, fontFace: BF, fontSize: 18, italic: true, color: "9FD8CE" });
s.addText([
  { text: "Best model: U-Net++ / U-Net   ", options: { color: WHITE, bold: true } },
  { text: "Test Dice 0.906  •  IoU 0.829", options: { color: MINT, bold: true } },
], { x: 0.9, y: 5.2, w: 11, h: 0.5, fontFace: BF, fontSize: 18 });
s.addText("Live demo: huggingface.co/spaces/unixio/skin-lesion-segmentation", { x: 0.9, y: 6.4,
  w: 11.6, h: 0.4, fontFace: BF, fontSize: 13, color: "9FD8CE" });

// ---- 2. PROBLEM & OBJECTIVES ---------------------------------------------
s = slide();
head(s, "Motivation", "The Problem & Our Objectives");
s.addText([
  { text: "Why it matters\n", options: { bold: true, fontSize: 17, color: TEAL } },
  { text: "Skin cancer survival depends on early detection. Outlining a lesion's exact boundary is needed to measure size, track growth, and plan treatment — but manual delineation is slow and subjective.",
    options: { fontSize: 15, color: INK } },
], { x: 0.6, y: 1.9, w: 6.0, h: 2.5, fontFace: BF, valign: "top", lineSpacingMultiple: 1.1 });
const objs = [
  ["1", "Implement U-Net for lesion segmentation"],
  ["2", "Compare with Attention U-Net & U-Net++"],
  ["3", "Evaluate with Dice, IoU & Hausdorff"],
  ["4", "Build an upload-image prototype"],
  ["5", "Explain predictions with Grad-CAM"],
];
let oy = 1.9;
objs.forEach(([n, t]) => {
  s.addShape("ellipse", { x: 7.0, y: oy, w: 0.5, h: 0.5, fill: { color: TEAL }, line: { type: "none" } });
  s.addText(n, { x: 7.0, y: oy, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: BF, fontSize: 16, bold: true, color: WHITE });
  s.addText(t, { x: 7.65, y: oy, w: 5.0, h: 0.5, valign: "middle", fontFace: BF, fontSize: 15, color: INK });
  oy += 0.72;
});

// ---- 3. WHAT IS SEGMENTATION ---------------------------------------------
s = slide(LIGHT);
head(s, "Concept", "What Is Semantic Segmentation?");
const cards = [
  ["Classification", "One label for the whole image", "“Is there a lesion?”", GRAY],
  ["Detection", "A box around the object", "“Roughly where?”", SEA],
  ["Segmentation", "A label for EVERY pixel", "“Exactly which pixels?”", TEAL],
];
let cx = 0.6;
cards.forEach(([t, d, q, col], i) => {
  chip(s, cx, 2.0, 3.9, 3.4, WHITE);
  s.addShape("rect", { x: cx, y: 2.0, w: 3.9, h: 0.18, fill: { color: col }, line: { type: "none" } });
  s.addText(t, { x: cx + 0.25, y: 2.4, w: 3.4, h: 0.6, fontFace: HF, fontSize: 20, bold: true, color: col });
  s.addText(d, { x: cx + 0.25, y: 3.1, w: 3.4, h: 1.0, fontFace: BF, fontSize: 15, color: INK, valign: "top" });
  s.addText(q, { x: cx + 0.25, y: 4.6, w: 3.4, h: 0.6, fontFace: BF, fontSize: 14, italic: true, color: GRAY });
  if (i === 2) s.addText("← OUR TASK", { x: cx + 1.9, y: 2.05, w: 1.9, h: 0.4, fontFace: BF, fontSize: 11, bold: true, color: WHITE, align: "right" });
  cx += 4.15;
});
s.addText("A supervised, pixel-wise binary classification: each pixel → lesion (1) or background (0).",
  { x: 0.6, y: 5.7, w: 12, h: 0.5, fontFace: BF, fontSize: 16, italic: true, bold: true, color: TEAL });

// ---- 4. DATASET -----------------------------------------------------------
s = slide();
head(s, "Data", "Dataset: ISIC-2018 Skin Lesions");
const stats = [["2,594", "image / mask pairs"], ["1,815", "training (70%)"], ["389", "validation (15%)"], ["390", "test (15%)"]];
let sx = 0.6;
stats.forEach(([big, lab]) => {
  chip(s, sx, 2.1, 2.9, 1.9, LIGHT);
  s.addText(big, { x: sx, y: 2.3, w: 2.9, h: 0.9, align: "center", fontFace: HF, fontSize: 40, bold: true, color: TEAL });
  s.addText(lab, { x: sx, y: 3.25, w: 2.9, h: 0.6, align: "center", fontFace: BF, fontSize: 14, color: GRAY });
  sx += 3.05;
});
s.addText([
  { text: "Expert-annotated", options: { bold: true, color: TEAL } },
  { text: " dermoscopy images, each with a ground-truth binary mask. Resized to 256×256. The lesion is the minority class (class imbalance) — which shapes our loss choice. Augmentation (flips, rotation, elastic deformation, CLAHE) improves generalisation on this small medical set.", options: { color: INK } },
], { x: 0.6, y: 4.4, w: 12.1, h: 2.0, fontFace: BF, fontSize: 16, valign: "top", lineSpacingMultiple: 1.15 });

// ---- 5. ARCHITECTURES -----------------------------------------------------
s = slide(LIGHT);
head(s, "Models", "Encoder–Decoder Architectures");
s.addText("Encoder shrinks the image to learn “what”; decoder upsamples to draw “where”; skip connections restore fine detail.",
  { x: 0.6, y: 1.75, w: 12.2, h: 0.6, fontFace: BF, fontSize: 15, italic: true, color: GRAY });
const arch = [
  ["U-Net", "ResNet-34 encoder, pretrained on ImageNet. Classic symmetric skip connections.", "~24.4M params", TEAL],
  ["Attention U-Net", "Adds attention gates on skips to suppress background. Trained from scratch.", "~31.4M params", SEA],
  ["U-Net++", "Dense nested skip pathways. Pretrained ResNet-34 encoder.", "~26.1M params", MINT],
];
let ax = 0.6;
arch.forEach(([t, d, pr, col]) => {
  chip(s, ax, 2.5, 3.9, 3.4, WHITE);
  s.addShape("ellipse", { x: ax + 0.3, y: 2.8, w: 0.7, h: 0.7, fill: { color: col }, line: { type: "none" } });
  s.addText(t, { x: ax + 0.25, y: 3.65, w: 3.5, h: 0.5, fontFace: HF, fontSize: 19, bold: true, color: INK });
  s.addText(d, { x: ax + 0.25, y: 4.2, w: 3.45, h: 1.4, fontFace: BF, fontSize: 13.5, color: INK, valign: "top" });
  s.addText(pr, { x: ax + 0.25, y: 5.45, w: 3.4, h: 0.4, fontFace: BF, fontSize: 13, bold: true, color: col });
  ax += 4.15;
});

// ---- 6. LOSS FUNCTIONS ----------------------------------------------------
s = slide();
head(s, "Training", "Loss Functions");
s.addText("Lesions are a minority of pixels — so we combine two losses to beat class imbalance.",
  { x: 0.6, y: 1.8, w: 12, h: 0.5, fontFace: BF, fontSize: 16, italic: true, color: GRAY });
const losses = [
  ["Binary Cross-Entropy", "Penalises each pixel independently", "BCE = -(1/N) Σ [ y·log ŷ + (1-y)·log(1-ŷ) ]"],
  ["Dice Loss", "Measures shape overlap — robust to imbalance", "Dice = 1 - (2 Σ ŷy + ε)/(Σŷ + Σy + ε)"],
  ["Combined (used)", "Pixel accuracy + shape accuracy", "L = 0.5·BCE + 0.5·Dice"],
];
let ly = 2.5;
losses.forEach(([t, d, f]) => {
  chip(s, 0.6, ly, 12.1, 1.25, LIGHT);
  s.addText(t, { x: 0.9, y: ly + 0.15, w: 4.0, h: 0.9, valign: "middle", fontFace: HF, fontSize: 18, bold: true, color: TEAL });
  s.addText(d, { x: 4.9, y: ly + 0.15, w: 3.6, h: 0.9, valign: "middle", fontFace: BF, fontSize: 13.5, color: INK });
  s.addText(f, { x: 8.5, y: ly + 0.15, w: 4.0, h: 0.9, valign: "middle", fontFace: "Consolas", fontSize: 12.5, italic: true, color: INK });
  ly += 1.4;
});

// ---- 7. METRICS -----------------------------------------------------------
s = slide(LIGHT);
head(s, "Evaluation", "Metrics — How We Judge Quality");
const mets = [
  ["Dice (DSC)", "2·TP / (2·TP + FP + FN)", "Primary overlap score (0–1)"],
  ["IoU / Jaccard", "TP / (TP + FP + FN)", "Standard overlap metric"],
  ["Precision / Recall", "TP/(TP+FP)  •  TP/(TP+FN)", "Recall = safety (no missed lesion)"],
  ["Hausdorff (HD95)", "max boundary distance (95%)", "Boundary accuracy for surgery"],
];
let mx = 0.6, my = 2.1;
mets.forEach(([t, f, d], i) => {
  const X = mx + (i % 2) * 6.15, Y = my + Math.floor(i / 2) * 2.0;
  chip(s, X, Y, 5.9, 1.7, WHITE);
  s.addText(t, { x: X + 0.3, y: Y + 0.2, w: 5.4, h: 0.5, fontFace: HF, fontSize: 18, bold: true, color: TEAL });
  s.addText(f, { x: X + 0.3, y: Y + 0.75, w: 5.4, h: 0.4, fontFace: "Consolas", fontSize: 13, italic: true, color: INK });
  s.addText(d, { x: X + 0.3, y: Y + 1.15, w: 5.4, h: 0.4, fontFace: BF, fontSize: 13, color: GRAY });
});

// ---- 8. RESULTS (table + chart) ------------------------------------------
s = slide();
head(s, "Results", "Test-Set Performance (390 images)");
const rows = [
  [{ text: "Model", options: { bold: true, color: WHITE, fill: { color: TEAL } } },
   { text: "Dice", options: { bold: true, color: WHITE, fill: { color: TEAL } } },
   { text: "IoU", options: { bold: true, color: WHITE, fill: { color: TEAL } } },
   { text: "Recall", options: { bold: true, color: WHITE, fill: { color: TEAL } } },
   { text: "HD95", options: { bold: true, color: WHITE, fill: { color: TEAL } } }],
  ["U-Net", "0.906", "0.829", "0.889", "12.46"],
  ["Attention U-Net", "0.882", "0.792", "0.854", "16.71"],
  ["U-Net++", "0.905", "0.829", "0.891", "12.80"],
];
s.addTable(rows, { x: 0.6, y: 2.0, w: 5.9, colW: [2.3, 0.9, 0.9, 0.9, 0.9], rowH: 0.6,
  fontFace: BF, fontSize: 14, align: "center", valign: "middle", border: { type: "solid", color: "DDDDDD", pt: 1 },
  color: INK });
s.addText("U-Net & U-Net++ are statistically tied; both use pretrained encoders.",
  { x: 0.6, y: 5.2, w: 5.9, h: 0.8, fontFace: BF, fontSize: 13, italic: true, color: GRAY, valign: "top" });
s.addImage({ path: path.join(A, "metrics_comparison.png"), x: 6.7, y: 1.9, w: 6.1, h: 3.39 });

// ---- 9. KEY FINDING -------------------------------------------------------
s = slide(NAVY);
s.addText("KEY FINDING", { x: 0.6, y: 1.0, w: 12, h: 0.5, fontFace: BF, fontSize: 15, bold: true, color: MINT, charSpacing: 3 });
s.addText("Transfer learning beat architectural complexity.", { x: 0.6, y: 1.6, w: 12.1, h: 1.2,
  fontFace: HF, fontSize: 34, bold: true, color: WHITE });
s.addText([
  { text: "The pretrained U-Net (0.906) matched the more complex U-Net++ (0.905) and clearly beat the scratch-trained Attention U-Net (0.882) — ", options: { color: "CFE8E3" } },
  { text: "despite Attention U-Net having the most parameters.", options: { color: WHITE, bold: true } },
], { x: 0.6, y: 3.1, w: 11.8, h: 1.4, fontFace: BF, fontSize: 19, valign: "top", lineSpacingMultiple: 1.15 });
const kf = [["0.906", "U-Net (pretrained)"], ["0.905", "U-Net++ (pretrained)"], ["0.882", "Attention (scratch)"]];
let kx = 0.6;
kf.forEach(([b, l]) => {
  s.addText(b, { x: kx, y: 4.9, w: 3.0, h: 0.9, fontFace: HF, fontSize: 40, bold: true, color: MINT, align: "center" });
  s.addText(l, { x: kx, y: 5.85, w: 3.0, h: 0.5, fontFace: BF, fontSize: 14, color: "CFE8E3", align: "center" });
  kx += 4.1;
});
s.addText("On small medical datasets, a pretrained encoder is the dominant lever for performance.",
  { x: 0.6, y: 6.6, w: 12, h: 0.5, fontFace: BF, fontSize: 15, italic: true, color: SEA });

// ---- 10. QUALITATIVE ------------------------------------------------------
s = slide();
head(s, "Results", "Qualitative Predictions (U-Net++)");
s.addImage({ path: path.join(A, "predictions_unetpp.png"), x: 3.4, y: 1.8, w: 6.3, h: 5.2 });
s.addText("Input  →  Expert ground truth  →  Predicted overlay", { x: 0.6, y: 6.9, w: 12, h: 0.4,
  align: "center", fontFace: BF, fontSize: 13, italic: true, color: GRAY });

// ---- 11. EXPLAINABILITY ---------------------------------------------------
s = slide(LIGHT);
head(s, "Extension", "Explainability with Grad-CAM");
s.addText([
  { text: "Why this region?\n", options: { bold: true, fontSize: 18, color: TEAL } },
  { text: "Grad-CAM produces a heatmap of the pixels that most pushed the model toward “lesion”.\n\n", options: { fontSize: 15, color: INK } },
  { text: "🔴 Warm = high influence\n", options: { fontSize: 15, color: INK, bold: true } },
  { text: "🔵 Cool = little influence\n\n", options: { fontSize: 15, color: INK, bold: true } },
  { text: "It lets a clinician verify the model focused on the lesion — not on hair, shadows, or the image border. Essential for trustworthy clinical AI.", options: { fontSize: 15, color: INK } },
], { x: 0.6, y: 2.0, w: 6.2, h: 4.5, fontFace: BF, valign: "top", lineSpacingMultiple: 1.1 });
chip(s, 7.2, 2.0, 5.5, 4.6, WHITE);
s.addText("From black box  →  inspectable tool", { x: 7.4, y: 2.3, w: 5.1, h: 0.6, fontFace: HF, fontSize: 18, bold: true, color: TEAL, align: "center" });
s.addText("The model turns a probability map into an explanation a human can audit, increasing trust and supporting safe clinical adoption.",
  { x: 7.5, y: 3.1, w: 4.9, h: 3.2, fontFace: BF, fontSize: 15, color: INK, valign: "top", align: "center", lineSpacingMultiple: 1.2 });

// ---- 12. PROTOTYPE --------------------------------------------------------
s = slide();
head(s, "Prototype", "Live Web Application");
const feats = [
  ["Upload", "Drag in any skin / dermoscopy photo"],
  ["Segment", "Mask + colour overlay in seconds"],
  ["Explain", "Grad-CAM heatmap on demand"],
  ["Summarise", "Area, confidence, border, disclaimer"],
];
let fy = 2.0;
feats.forEach(([t, d]) => {
  s.addShape("roundRect", { x: 0.6, y: fy, w: 6.0, h: 0.95, fill: { color: LIGHT }, line: { type: "none" }, rectRadius: 0.06 });
  s.addText(t, { x: 0.85, y: fy, w: 1.8, h: 0.95, valign: "middle", fontFace: HF, fontSize: 17, bold: true, color: TEAL });
  s.addText(d, { x: 2.6, y: fy, w: 3.9, h: 0.95, valign: "middle", fontFace: BF, fontSize: 14, color: INK });
  fy += 1.1;
});
chip(s, 7.0, 2.0, 5.7, 4.4, NAVY);
s.addText("Deployed & always-on", { x: 7.2, y: 2.3, w: 5.3, h: 0.6, fontFace: HF, fontSize: 20, bold: true, color: WHITE, align: "center" });
s.addText([
  { text: "Hugging Face Spaces (free CPU)\n\n", options: { color: MINT, bold: true, fontSize: 15 } },
  { text: "Models pulled from the HF Hub at runtime — no local install, accessible any time.\n\n", options: { color: "CFE8E3", fontSize: 14 } },
  { text: "huggingface.co/spaces/\nunixio/skin-lesion-segmentation", options: { color: WHITE, fontSize: 13, italic: true } },
], { x: 7.3, y: 3.1, w: 5.1, h: 3.1, fontFace: BF, valign: "top", align: "center", lineSpacingMultiple: 1.15 });

// ---- 13. CLINICAL RELEVANCE ----------------------------------------------
s = slide(LIGHT);
head(s, "Impact", "Clinical Relevance");
const cl = [
  ["Early screening", "Flag suspicious lesions for review, cutting diagnostic delay", SEA],
  ["Size tracking", "Objective mask area to monitor lesion growth over time", TEAL],
  ["Surgical planning", "Pixel-precise boundaries (HD95) inform excision margins", MINT],
];
let clx = 0.6;
cl.forEach(([t, d, col]) => {
  chip(s, clx, 2.1, 3.9, 2.6, WHITE);
  s.addShape("rect", { x: clx, y: 2.1, w: 3.9, h: 0.16, fill: { color: col }, line: { type: "none" } });
  s.addText(t, { x: clx + 0.25, y: 2.45, w: 3.4, h: 0.6, fontFace: HF, fontSize: 19, bold: true, color: INK });
  s.addText(d, { x: clx + 0.25, y: 3.15, w: 3.45, h: 1.4, fontFace: BF, fontSize: 14.5, color: INK, valign: "top" });
  clx += 4.15;
});
s.addText([
  { text: "Melanoma 5-year survival: ", options: { color: INK } },
  { text: "~98% at Stage I", options: { color: TEAL, bold: true } },
  { text: "  vs  ", options: { color: INK } },
  { text: "~23% at Stage IV", options: { color: "B23A48", bold: true } },
  { text: "  — early, accurate detection saves lives.", options: { color: INK } },
], { x: 0.6, y: 5.3, w: 12.1, h: 0.8, fontFace: BF, fontSize: 18, align: "center", valign: "middle" });

// ---- 14. CONCLUSION -------------------------------------------------------
s = slide(NAVY);
s.addShape("rect", { x: 0, y: 0, w: 0.35, h: H, fill: { color: TEAL } });
s.addText("CONCLUSION", { x: 0.9, y: 1.0, w: 12, h: 0.5, fontFace: BF, fontSize: 15, bold: true, color: MINT, charSpacing: 3 });
s.addText("A complete, deployed segmentation system.", { x: 0.9, y: 1.55, w: 11.6, h: 1.0,
  fontFace: HF, fontSize: 32, bold: true, color: WHITE });
const concl = [
  "3 encoder–decoder models trained & compared on ISIC-2018",
  "Best test Dice 0.906 — on par with published benchmarks",
  "Full metrics: Dice, IoU, Precision, Recall, Hausdorff",
  "Live web prototype with Grad-CAM explainability",
  "Key insight: transfer learning > architectural complexity",
];
let yy = 2.9;
concl.forEach((t) => {
  s.addShape("ellipse", { x: 0.95, y: yy + 0.05, w: 0.28, h: 0.28, fill: { color: MINT }, line: { type: "none" } });
  s.addText(t, { x: 1.45, y: yy - 0.1, w: 11, h: 0.55, valign: "middle", fontFace: BF, fontSize: 18, color: "EAF4F2" });
  yy += 0.72;
});
s.addText("Code: github.com/BertinAm/unet-skin-lesion-segmentation   •   Demo: huggingface.co/spaces/unixio/skin-lesion-segmentation",
  { x: 0.9, y: 6.7, w: 12, h: 0.4, fontFace: BF, fontSize: 12, color: "9FD8CE" });

p.writeFile({ fileName: path.join(__dirname, "Medical_Image_Segmentation_Slides.pptx") })
  .then((f) => console.log("DECK_WRITTEN", f));
