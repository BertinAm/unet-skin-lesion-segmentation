/* Builds the project report DOCX. Run: NODE_PATH=$(npm root -g) node report/build_report.js */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak, TableOfContents,
  ExternalHyperlink, TabStopType, TabStopPosition,
} = require("docx");

const ASSETS = path.join(__dirname, "assets");
const img = (f) => fs.readFileSync(path.join(ASSETS, f));

const BLUE = "1F4E79", LGRAY = "F2F2F2", HEAD = "D5E8F0";
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border,
  insideHorizontal: border, insideVertical: border };

// ---- helpers ---------------------------------------------------------------
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const P = (runs, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED,
  children: Array.isArray(runs) ? runs : [new TextRun(runs)], ...opts });
const T = (t, o = {}) => new TextRun({ text: t, ...o });
const bullet = (runs) => new Paragraph({ numbering: { reference: "bullets", level: 0 },
  spacing: { after: 60 }, children: Array.isArray(runs) ? runs : [new TextRun(runs)] });
const num = (runs) => new Paragraph({ numbering: { reference: "numbers", level: 0 },
  spacing: { after: 60 }, children: Array.isArray(runs) ? runs : [new TextRun(runs)] });
const formula = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
  children: [new TextRun({ text: t, italics: true, font: "Cambria Math", size: 24 })] });
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: t, italics: true, size: 18, color: "555555" })] });
const figure = (file, w, h, cap) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
    children: [new ImageRun({ type: "png", data: img(file),
      transformation: { width: w, height: h },
      altText: { title: cap, description: cap, name: file } })] }),
  caption(cap),
];

function cell(text, { head = false, w = 1872, bold = false, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: { fill: head ? HEAD : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align,
      children: [new TextRun({ text, bold: head || bold, size: 20 })] })],
  });
}
function table(headers, rows, widths) {
  const mk = (cells, head) => new TableRow({
    children: cells.map((c, i) => cell(String(c), { head, w: widths[i],
      align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) });
  return new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths, rows: [mk(headers, true), ...rows.map((r) => mk(r, false))] });
}

// ===========================================================================
const children = [];

// ---- Title page ----
children.push(
  new Paragraph({ spacing: { before: 2400, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Automated Medical Image Segmentation System",
      bold: true, size: 48, color: BLUE })] }),
  new Paragraph({ spacing: { before: 120, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Deep Learning-Based Segmentation of Skin Lesions Using U-Net Variants",
      size: 30, color: "333333" })] }),
  new Paragraph({ spacing: { before: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "U-Net  •  Attention U-Net  •  U-Net++   on the ISIC-2018 dataset",
      italics: true, size: 24, color: "666666" })] }),
  new Paragraph({ spacing: { before: 1400 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Project Report", bold: true, size: 28 })] }),
  new Paragraph({ spacing: { before: 80 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Theme: Image Segmentation & Model Optimization", size: 22 })] }),
  new Paragraph({ spacing: { before: 1200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Live demo, code and trained models", bold: true, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60 },
    children: [new ExternalHyperlink({ link: "https://huggingface.co/spaces/unixio/skin-lesion-segmentation",
      children: [new TextRun({ text: "Live App (Hugging Face Spaces)", style: "Hyperlink", size: 20 })] })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40 },
    children: [new ExternalHyperlink({ link: "https://github.com/BertinAm/unet-skin-lesion-segmentation",
      children: [new TextRun({ text: "Source Code (GitHub)", style: "Hyperlink", size: 20 })] })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40 },
    children: [new ExternalHyperlink({ link: "https://huggingface.co/unixio/unet-skin-lesion-segmentation",
      children: [new TextRun({ text: "Trained Models (Hugging Face Hub)", style: "Hyperlink", size: 20 })] })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---- TOC ----
children.push(
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Table of Contents")] }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---- Abstract ----
children.push(H1("Abstract"));
children.push(P([
  T("This project develops an "), T("automated medical image segmentation system", { bold: true }),
  T(" that identifies skin lesions in dermoscopy images and produces a pixel-level segmentation mask. "),
  T("Three encoder–decoder architectures — the baseline "), T("U-Net", { bold: true }),
  T(", "), T("Attention U-Net", { bold: true }), T(", and "), T("U-Net++", { bold: true }),
  T(" — were implemented and compared on the ISIC-2018 lesion-segmentation dataset (2,594 image/mask pairs). "),
  T("Models were trained with a combined Binary Cross-Entropy + Dice loss and evaluated with Dice, IoU, "),
  T("Precision, Recall and the 95th-percentile Hausdorff distance on a held-out test set. "),
  T("The best model (U-Net) reached a "), T("test Dice of 0.906 and IoU of 0.829", { bold: true }),
  T(", matching published benchmarks. A Gradio web application lets users upload an image and visualise the "),
  T("predicted mask, and "), T("Grad-CAM", { bold: true }),
  T(" explainability highlights the regions driving each prediction. The system is deployed publicly and runs without any local installation."),
]));

// ---- 1. Introduction ----
children.push(H1("1. Introduction"));
children.push(P([
  T("Medical image analysis is one of the highest-impact applications of computer vision. "),
  T("Skin cancer — particularly melanoma — is among the most common cancers worldwide; survival depends heavily on "),
  T("early detection. Accurately delineating a lesion's boundary is a prerequisite for measuring its size, tracking "),
  T("its growth over time, and planning excision margins. Manual delineation is slow, subjective and prone to "),
  T("inter-observer variability, which motivates automated segmentation."),
]));
children.push(H2("1.1 Objectives"));
children.push(num("Implement semantic segmentation of skin lesions using the U-Net architecture."));
children.push(num("Compare U-Net against two advanced variants — Attention U-Net and U-Net++."));
children.push(num("Evaluate all models with clinically meaningful metrics (Dice, IoU, Hausdorff distance)."));
children.push(num("Deliver a prototype where a user uploads an image and visualises the predicted mask."));
children.push(num("Apply explainability (Grad-CAM) to reveal why regions were segmented (extension)."));

// ---- 2. Dataset ----
children.push(H1("2. Dataset"));
children.push(P([
  T("We use the "), T("ISIC-2018 Task 1", { bold: true }),
  T(" (Lesion Boundary Segmentation) dataset from the International Skin Imaging Collaboration. "),
  T("It contains "), T("2,594 dermoscopy images", { bold: true }),
  T(", each paired with an expert-annotated binary ground-truth mask (lesion vs. background). "),
  T("Images were resized to 256×256 and split into training, validation and test sets:"),
]));
children.push(table(
  ["Split", "Images", "Proportion", "Purpose"],
  [["Train", "1,815", "70%", "Learn model weights"],
   ["Validation", "389", "15%", "Tune / select best epoch"],
   ["Test", "390", "15%", "Final unbiased evaluation"]],
  [2340, 2340, 2340, 2340]));
children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
children.push(P([
  T("Because the foreground (lesion) is the minority class, the data exhibits class imbalance, which directly "),
  T("informs the choice of loss function (Section 5). Data augmentation — flips, rotations, elastic deformation, "),
  T("brightness/contrast jitter and CLAHE — was applied during training to improve generalisation on this "),
  T("relatively small medical dataset."),
]));

// ---- 3. Semantic segmentation principles ----
children.push(H1("3. Semantic Segmentation Principles"));
children.push(P([
  T("Image tasks differ by the granularity of their output. "),
  T("Classification", { bold: true }), T(" assigns one label to the whole image; "),
  T("object detection", { bold: true }), T(" draws bounding boxes; "),
  T("semantic segmentation", { bold: true }),
  T(" assigns a class to "), T("every pixel", { italics: true }),
  T(". Lesion segmentation is therefore a "), T("pixel-wise binary classification", { bold: true }),
  T(" problem: each pixel is predicted as lesion (1) or background (0), yielding a mask the same size as the input."),
]));
children.push(P([
  T("This is a "), T("supervised learning", { bold: true }),
  T(" problem — the model learns from images paired with expert masks. The network outputs a probability map; "),
  T("applying a threshold (0.5) converts it into the final binary mask. The central difficulty is producing "),
  T("spatially precise boundaries, which is exactly what the encoder–decoder design addresses."),
]));

// ---- 4. Encoder-decoder architectures ----
children.push(H1("4. Encoder–Decoder Architectures"));
children.push(P([
  T("All three models follow the encoder–decoder paradigm. The "), T("encoder", { bold: true }),
  T(" progressively downsamples the image, extracting increasingly abstract features (\"what\" is present) while "),
  T("losing spatial resolution. The "), T("decoder", { bold: true }),
  T(" upsamples these features back to the original resolution to produce the mask (\"where\" it is). "),
  T("Skip connections", { bold: true }),
  T(" bridge encoder and decoder, restoring fine spatial detail lost during downsampling."),
]));
children.push(H3("4.1 Core operations"));
children.push(P([T("Convolution", { bold: true }), T(" detects local patterns (edges, colour, texture):")]));
children.push(formula("Output(i,j) = Σm Σn  Input(i+m, j+n) × Filter(m,n)"));
children.push(P([T("Max pooling", { bold: true }), T(" halves spatial size, keeping the strongest activation:")]));
children.push(formula("Pool(i,j) = max( region around (i,j) )"));
children.push(P([T("Transposed convolution", { bold: true }), T(" reverses pooling to upsample in the decoder.")]));

children.push(H3("4.2 U-Net (baseline)"));
children.push(P([
  T("The classic U-Net has a symmetric \"U\" shape: a contracting encoder, an expanding decoder, and skip "),
  T("connections that concatenate matching-resolution feature maps. We use a "),
  T("ResNet-34 encoder pretrained on ImageNet", { bold: true }),
  T(" (transfer learning), which gives strong features even with limited medical data (~24.4M parameters)."),
]));
children.push(H3("4.3 Attention U-Net"));
children.push(P([
  T("Attention U-Net adds "), T("attention gates", { bold: true }),
  T(" on the skip connections. Each gate learns to suppress irrelevant background activations and emphasise the "),
  T("lesion, using a coarse gating signal g from the decoder and fine features x from the encoder:"),
]));
children.push(formula("q = ReLU(Wg·g + Wx·x);   ψ = σ(Wψ·q);   output = x · ψ"));
children.push(P([
  T("The coefficients ψ ∈ [0,1] re-weight the skip features. In this study the Attention U-Net was trained "),
  T("from scratch (no pretrained encoder), totalling ~31.4M parameters."),
]));
children.push(H3("4.4 U-Net++"));
children.push(P([
  T("U-Net++ redesigns the skip pathways with a "), T("dense nest of intermediate convolution blocks", { bold: true }),
  T(", reducing the semantic gap between encoder and decoder features before they are fused. It also uses the "),
  T("pretrained ResNet-34 encoder (~26.1M parameters)."),
]));

// ---- 5. Loss functions ----
children.push(H1("5. Loss Functions"));
children.push(P([
  T("A loss function measures how wrong a prediction is; training minimises it. Because lesions occupy a minority "),
  T("of pixels, a naïve loss can be \"fooled\" by predicting all-background, so we combine two complementary losses."),
]));
children.push(H3("5.1 Binary Cross-Entropy (BCE)"));
children.push(P("BCE penalises each pixel independently — large penalty when the model is confidently wrong:"));
children.push(formula("BCE = -(1/N) Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]"));
children.push(H3("5.2 Dice Loss"));
children.push(P("Dice loss measures overlap of shapes, so it is robust to class imbalance:"));
children.push(formula("Dice Loss = 1 - (2 Σ ŷ·y + ε) / (Σ ŷ + Σ y + ε)"));
children.push(H3("5.3 Combined loss (used)"));
children.push(P("BCE catches pixel-level errors; Dice catches shape-level errors. We weight them equally:"));
children.push(formula("L = 0.5 · BCE + 0.5 · Dice Loss"));
children.push(P([
  T("A "), T("Tversky loss", { bold: true }),
  T(" (α=0.7, β=0.3) is also provided for cases needing stronger penalties on false negatives (missed lesion)."),
]));

// ---- 6. Evaluation metrics ----
children.push(H1("6. Evaluation Metrics"));
children.push(P([T("Let TP, FP, FN be true positives, false positives and false negatives at the pixel level.")]));
children.push(H3("6.1 Dice Score (DSC)"));
children.push(P("Primary metric — harmonic overlap between prediction and ground truth (0–1, higher better):"));
children.push(formula("Dice = 2·TP / (2·TP + FP + FN) = 2|A∩B| / (|A| + |B|)"));
children.push(H3("6.2 Intersection over Union (IoU / Jaccard)"));
children.push(formula("IoU = TP / (TP + FP + FN) = |A∩B| / |A∪B|"));
children.push(P("Dice and IoU are related by  Dice = 2·IoU / (1 + IoU); Dice is always ≥ IoU."));
children.push(H3("6.3 Precision and Recall"));
children.push(formula("Precision = TP / (TP + FP)        Recall = TP / (TP + FN)"));
children.push(P([
  T("In a clinical screening context "), T("recall", { italics: true }),
  T(" (sensitivity) is critical — missing real lesion tissue is more dangerous than a false alarm."),
]));
children.push(H3("6.4 Hausdorff Distance (HD95)"));
children.push(P([
  T("Dice/IoU measure overlap but not boundary quality. The 95th-percentile Hausdorff distance measures the "),
  T("worst-case (robustly, ignoring the top 5% outliers) distance between predicted and true boundaries — "),
  T("clinically important for excision margins. Lower is better."),
]));
children.push(formula("H(A,B) = max( sup_a inf_b d(a,b),  sup_b inf_a d(a,b) )"));

// ---- 7. Methodology ----
children.push(H1("7. Methodology"));
children.push(P("All experiments used identical data splits and training configuration for a fair comparison:"));
children.push(table(
  ["Setting", "Value"],
  [["Framework", "PyTorch + segmentation-models-pytorch"],
   ["Image size", "256 × 256"],
   ["Optimizer", "AdamW (lr 3e-4, weight decay 1e-4)"],
   ["Scheduler", "Cosine annealing with warm restarts (T₀=10)"],
   ["Precision", "Mixed precision (AMP) on GPU"],
   ["Loss", "0.5·BCE + 0.5·Dice"],
   ["Max epochs", "60, early stopping (patience 15)"],
   ["Hardware", "NVIDIA GPU (Kaggle)"]],
  [3120, 6240]));
children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
children.push(P([
  T("The best checkpoint was selected by validation Dice. U-Net and U-Net++ used ImageNet-pretrained ResNet-34 "),
  T("encoders; Attention U-Net was trained from scratch."),
]));

// ---- 8. Results ----
children.push(H1("8. Results and Discussion"));
children.push(P("Final metrics on the held-out 390-image test set:"));
children.push(table(
  ["Model", "Dice ↑", "IoU ↑", "Precision ↑", "Recall ↑", "HD95 ↓"],
  [["U-Net (ResNet-34)", "0.906", "0.829", "0.926", "0.889", "12.46"],
   ["Attention U-Net", "0.882", "0.792", "0.919", "0.854", "16.71"],
   ["U-Net++ (ResNet-34)", "0.905", "0.829", "0.924", "0.891", "12.80"]],
  [3360, 1200, 1200, 1320, 1200, 1080]));
children.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
children.push(...figure("metrics_comparison.png", 540, 300, "Figure 1. Dice, IoU, Precision and Recall across the three models on the ISIC-2018 test set."));
children.push(...figure("hd95_comparison.png", 360, 270, "Figure 2. Boundary accuracy (HD95, lower is better). U-Net and U-Net++ produce tighter boundaries than Attention U-Net."));
children.push(H2("8.1 Key findings"));
children.push(bullet([T("U-Net and U-Net++ are statistically tied", { bold: true }),
  T(" (0.906 vs 0.905 Dice — a 0.0004 difference). The added complexity of U-Net++ gave no measurable benefit on this dataset.")]));
children.push(bullet([T("Attention U-Net scored lowest", { bold: true }),
  T(" (0.882) despite having the most parameters. The decisive factor was transfer learning: U-Net and U-Net++ used ImageNet-pretrained encoders, whereas Attention U-Net was trained from scratch.")]));
children.push(bullet([T("Lesson: ", { bold: true }),
  T("on a ~1,800-image dataset, a pretrained encoder matters more than architectural sophistication.")]));
children.push(bullet([T("Boundary quality (HD95) agrees with Dice — the pretrained models give ~12.5 px boundary error vs 16.7 px for the scratch-trained model.")]));
children.push(bullet([T("All models are slightly conservative (precision > recall), occasionally missing lesion edges rather than over-segmenting.")]));
children.push(...figure("predictions_unetpp.png", 430, 377, "Figure 3. Qualitative U-Net++ predictions on test images (input | ground truth | predicted overlay). Predicted masks closely follow expert annotations."));

// ---- 9. Explainability ----
children.push(H1("9. Explainability (Extension): Grad-CAM"));
children.push(P([
  T("To address the project's extension, we apply "), T("Grad-CAM", { bold: true }),
  T(" (Gradient-weighted Class Activation Mapping). For segmentation we define the target as the total predicted "),
  T("lesion probability, so the heatmap highlights the pixels that most increased the model's belief that a lesion "),
  T("is present. Warm colours (red/orange) indicate high influence; cool colours (blue) indicate little influence."),
]));
children.push(P([
  T("This turns the model from a black box into an inspectable tool: a clinician can verify the network focused on "),
  T("the actual lesion rather than on hair, shadows, ink markings or the image border — a key requirement for "),
  T("trustworthy clinical AI."),
]));

// ---- 10. Prototype ----
children.push(H1("10. Prototype Application"));
children.push(P([
  T("The required prototype is a "), T("Gradio web application", { bold: true }),
  T(" deployed on Hugging Face Spaces. A user uploads any skin image; the app returns the predicted mask, a "),
  T("colour overlay, a Grad-CAM heatmap, and a written summary (lesion area, model confidence, border "),
  T("description, and a clinical disclaimer). Trained weights are pulled from the Hugging Face model hub at "),
  T("runtime, so the app needs no local installation and is accessible at any time."),
]));
children.push(P([T("Live application: ", { bold: true }),
  new ExternalHyperlink({ link: "https://huggingface.co/spaces/unixio/skin-lesion-segmentation",
    children: [new TextRun({ text: "huggingface.co/spaces/unixio/skin-lesion-segmentation", style: "Hyperlink" })] })]));
children.push(P([
  T("In informal testing the model — trained only on dermoscopy close-ups — successfully localised affected skin "),
  T("on an ordinary phone photo, demonstrating generalisation beyond its training distribution, while the summary "),
  T("transparently flags that boundaries on wide photos should be treated as approximate."),
]));

// ---- 11. Clinical relevance ----
children.push(H1("11. Clinical Relevance"));
children.push(P([
  T("Accurate lesion segmentation supports several real clinical workflows:"),
]));
children.push(bullet([T("Early screening: ", { bold: true }), T("automatically flag suspicious lesions for closer review, reducing diagnostic delay.")]));
children.push(bullet([T("Size tracking: ", { bold: true }), T("precise mask area enables objective monitoring of lesion growth across visits.")]));
children.push(bullet([T("Surgical planning: ", { bold: true }), T("pixel-precise boundaries (validated by HD95) inform excision margins.")]));
children.push(P([
  T("Caught early (Stage I), melanoma has ~98% five-year survival versus ~23% at Stage IV — so tools that aid "),
  T("early, accurate detection carry significant clinical value. The metric priorities reflect this: recall guards "),
  T("against missed disease, while Hausdorff distance guards boundary accuracy for treatment."),
]));

// ---- 12. Limitations ----
children.push(H1("12. Limitations and Future Work"));
children.push(bullet("Single dataset (ISIC-2018) — external validation on other cohorts is needed."));
children.push(bullet("Domain shift: precision drops on consumer-camera photos vs. dermoscopy close-ups."));
children.push(bullet("Binary segmentation only — it localises lesions but does not classify benign vs. malignant."));
children.push(bullet([T("Future work: ", { bold: true }),
  T("add a classification head, give Attention U-Net a pretrained encoder for a fairer comparison, explore transformer variants (TransUNet/Swin-UNet), and push recall higher for safety-critical screening.")]));

// ---- 13. Conclusion ----
children.push(H1("13. Conclusion"));
children.push(P([
  T("We built an end-to-end skin-lesion segmentation system: three encoder–decoder models trained and compared "),
  T("on ISIC-2018, evaluated with Dice, IoU, Precision, Recall and Hausdorff distance, wrapped in a deployed "),
  T("web prototype with Grad-CAM explainability. The best model reaches a "),
  T("test Dice of 0.906", { bold: true }),
  T(", on par with published results. The central finding — that a pretrained U-Net matches the more complex "),
  T("U-Net++ and beats a scratch-trained Attention U-Net — highlights that, for small medical datasets, "),
  T("transfer learning is the dominant lever for performance."),
]));

// ---- References ----
children.push(H1("References"));
const refs = [
  "Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.",
  "Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. MIDL.",
  "Zhou, Z., et al. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. DLMIA.",
  "Codella, N., et al. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018 (ISIC). arXiv:1902.03368.",
  "Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.",
  "Milletari, F., et al. (2016). V-Net / Dice loss for volumetric medical image segmentation. 3DV.",
];
refs.forEach((r) => children.push(num(r)));

// ===========================================================================
const doc = new Document({
  creator: "unixio",
  title: "Automated Medical Image Segmentation System",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: BLUE, font: "Arial" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: "2E5E8C", font: "Arial" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "333333", font: "Arial" },
        paragraph: { spacing: { before: 140, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "Skin Lesion Segmentation — Project Report", size: 16, color: "999999" })],
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 4 } } })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Page ", size: 16, color: "999999" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "999999" }),
        new TextRun({ text: " of ", size: 16, color: "999999" }),
        new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: "999999" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Medical_Image_Segmentation_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("REPORT_WRITTEN", out, buf.length, "bytes");
});
