import type { jsPDF as JsPdf } from 'jspdf';
import type { StudyMetadata } from '../dicom/types';
import type { ChatMessage, ResolvedCircleAnnotation, SelectionPlan } from '../llm/types';
import type { AnnotatedSliceImage } from './sliceImage';
import type { ExportFinding, ExportOptions, ExportReport, FindingTier } from './types';

export interface PdfBuildInput {
  report: ExportReport;
  metadata: StudyMetadata;
  findings: ResolvedCircleAnnotation[];
  /** uid → annotated slice image. */
  images: Map<string, AnnotatedSliceImage>;
  messages: ChatMessage[];
  plan?: SelectionPlan | null;
  options: ExportOptions;
  /** e.g. "Claude · claude-opus-4-8" or "Fallback (deterministic)". */
  providerLabel: string;
  generatedAt: Date;
}

// --- Layout constants (mm) ---
const PAGE_W = 210;
const PAGE_H = 297;
const MARGIN = 16;
const CONTENT_W = PAGE_W - MARGIN * 2;
const FOOTER_Y = PAGE_H - 10;

type RGB = [number, number, number];
const INK: RGB = [31, 41, 55];
const MUTED: RGB = [107, 114, 128];
const ACCENT: RGB = [29, 78, 216];
const ACCENT_DEEP: RGB = [30, 58, 138];
const LINE: RGB = [226, 232, 240];
const PANEL: RGB = [249, 250, 251];
const WHITE: RGB = [255, 255, 255];
const WARN_BG: RGB = [254, 242, 242];
const WARN_BORDER: RGB = [252, 165, 165];
const WARN_TEXT: RGB = [185, 28, 28];

function tierColor(tier?: FindingTier): RGB {
  if (tier === 'definite') return [220, 38, 38];
  if (tier === 'probable') return [217, 119, 6];
  return [107, 114, 128];
}

function formatStudyDate(raw?: string): string {
  if (!raw) return '—';
  const m = raw.match(/^(\d{4})(\d{2})(\d{2})$/);
  return m ? `${m[1]}-${m[2]}-${m[3]}` : raw;
}

/** Strip the app's markdown-ish tokens for a clean plain-text transcript. */
function stripMarkdown(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, '')
    .split('\n')
    .map((ln) =>
      ln
        .replace(/^#{1,6}\s+/, '')
        .replace(/^\s*[-•*]\s+/, '• ')
        .replace(/\*\*(.*?)\*\*/g, '$1')
        .trimEnd(),
    )
    .join('\n')
    .trim();
}

/**
 * Build the export PDF and return it as a Blob. jsPDF is imported dynamically so
 * it is code-split out of the main app bundle (only loaded when a user exports).
 */
export async function buildReportPdf(input: PdfBuildInput): Promise<Blob> {
  const { jsPDF } = await import('jspdf');
  const doc = new jsPDF({ unit: 'mm', format: 'a4', compress: true });
  const b = new Builder(doc);

  b.headerBand(input.providerLabel, input.generatedAt);
  b.title(input.report.title);
  b.disclaimerBanner();
  b.studyInfo(input.metadata, input.findings.length, input.providerLabel);

  b.section('Indication');
  b.paragraph(input.report.indication);

  b.section('Technique');
  b.paragraph(input.report.technique);

  b.section('Impression');
  b.paragraph(input.report.impression, { size: 10.5, style: 'normal' });

  // Findings — attach an annotated image to each where one is available.
  b.section(`Findings (${input.report.findings.length})`);
  if (input.report.findings.length === 0) {
    b.paragraph('No discrete findings were reported for this review.', { color: MUTED, style: 'italic' });
  } else {
    const used = new Set<string>();
    input.report.findings.forEach((f, i) => {
      const ann = input.options.includeImages
        ? matchAnnotation(f, input.findings, used)
        : undefined;
      const img = ann ? input.images.get(ann.uid) : undefined;
      b.findingBlock(f, i + 1, img);
    });

    // Any marked regions the report didn't tie to a finding → a small gallery.
    if (input.options.includeImages) {
      const leftovers = input.findings.filter((a) => !used.has(a.uid) && input.images.has(a.uid));
      if (leftovers.length > 0) {
        b.subheading('Other marked regions');
        for (const a of leftovers) {
          const img = input.images.get(a.uid)!;
          b.image(img, `Series #${a.seriesNumber}, Slice ${a.instanceNumber} — ${a.label}`);
        }
      }
    }
  }

  b.section('Recommendations');
  b.paragraph(input.report.recommendations);

  b.section('Limitations');
  b.paragraph(input.report.limitations, { size: 9, color: MUTED });

  if (input.options.includeSelectionRationale && input.plan) {
    b.section('AI Slice Selection');
    b.paragraph(input.plan.reasoning, { size: 9 });
    for (const sel of input.plan.selections) {
      b.paragraph(`• Series #${sel.seriesNumber} (${sel.role}): ${sel.rationale}`, {
        size: 8.5,
        color: MUTED,
        indent: 3,
      });
    }
  }

  if (input.options.includeTranscript && input.messages.length > 0) {
    b.section('Conversation Transcript');
    for (const msg of input.messages) {
      b.transcriptEntry(msg);
    }
  }

  b.finishFooters();
  return doc.output('blob');
}

/** Best-effort match of a report finding to a marked annotation (by slice, then label). */
function matchAnnotation(
  finding: ExportFinding,
  annotations: ResolvedCircleAnnotation[],
  used: Set<string>,
): ResolvedCircleAnnotation | undefined {
  const free = annotations.filter((a) => !used.has(a.uid));
  // 1) exact series + instance
  if (finding.seriesNumber && finding.instanceNumber != null) {
    const hit = free.find(
      (a) => a.seriesNumber === finding.seriesNumber && a.instanceNumber === finding.instanceNumber,
    );
    if (hit) { used.add(hit.uid); return hit; }
  }
  // 2) label overlap — only within the finding's series (when it has one), and
  // requiring a meaningful (≥4 char) overlap so a short incidental substring
  // can't tie a finding to an unrelated region.
  const title = finding.title.trim().toLowerCase();
  if (title.length >= 4) {
    const hit = free.find((a) => {
      if (finding.seriesNumber && a.seriesNumber !== finding.seriesNumber) return false;
      const l = a.label.trim().toLowerCase();
      if (l.length < 4) return false;
      return l.includes(title) || title.includes(l);
    });
    if (hit) { used.add(hit.uid); return hit; }
  }
  return undefined;
}

interface ParagraphOpts {
  size?: number;
  color?: RGB;
  style?: 'normal' | 'bold' | 'italic';
  gapAfter?: number;
  indent?: number;
}

/** Stateful layout helper over a jsPDF doc: tracks the y-cursor and paginates. */
class Builder {
  private y = MARGIN;
  private doc: JsPdf;
  constructor(doc: JsPdf) {
    this.doc = doc;
  }

  private fill(c: RGB) { this.doc.setFillColor(c[0], c[1], c[2]); }
  private stroke(c: RGB) { this.doc.setDrawColor(c[0], c[1], c[2]); }
  private text(c: RGB) { this.doc.setTextColor(c[0], c[1], c[2]); }

  private newPage() { this.doc.addPage(); this.y = MARGIN; }
  private ensure(h: number) { if (this.y + h > FOOTER_Y - 5) this.newPage(); }

  headerBand(providerLabel: string, generatedAt: Date) {
    this.fill(ACCENT_DEEP);
    this.doc.rect(0, 0, PAGE_W, 24, 'F');
    this.text(WHITE);
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(15);
    this.doc.text('DICOMassist', MARGIN, 11);
    this.doc.setFont('helvetica', 'normal');
    this.doc.setFontSize(9);
    this.doc.text('AI-Assisted Imaging Report', MARGIN, 17);
    this.doc.setFontSize(8);
    this.doc.text(generatedAt.toLocaleString(), PAGE_W - MARGIN, 11, { align: 'right' });
    this.doc.text(providerLabel, PAGE_W - MARGIN, 17, { align: 'right' });
    this.y = 31;
  }

  title(t: string) {
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(14);
    this.text(INK);
    const lines = this.doc.splitTextToSize(t, CONTENT_W);
    for (const ln of lines) {
      this.ensure(7);
      this.doc.text(ln, MARGIN, this.y);
      this.y += 6.5;
    }
    this.y += 1.5;
  }

  disclaimerBanner() {
    const h = 13;
    this.ensure(h + 3);
    this.fill(WARN_BG);
    this.stroke(WARN_BORDER);
    this.doc.setLineWidth(0.3);
    this.doc.rect(MARGIN, this.y, CONTENT_W, h, 'FD');
    this.text(WARN_TEXT);
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(8.5);
    this.doc.text('NOT FOR CLINICAL USE', MARGIN + 3, this.y + 5);
    this.doc.setFont('helvetica', 'normal');
    this.doc.setFontSize(7.5);
    const sub = this.doc.splitTextToSize(
      'Research/educational tool. AI output may be incomplete or inaccurate. All findings must be confirmed by a qualified radiologist.',
      CONTENT_W - 6,
    );
    this.doc.text(sub, MARGIN + 3, this.y + 9);
    this.y += h + 4;
  }

  studyInfo(metadata: StudyMetadata, findingsCount: number, providerLabel: string) {
    const pairs: Array<[string, string]> = [
      ['Study', metadata.studyDescription || '—'],
      ['Modality', metadata.modality || '—'],
      ['Study Date', formatStudyDate(metadata.studyDate)],
      ['Patient', [metadata.patientSex, metadata.patientAge].filter(Boolean).join(', ') || '—'],
      ['Institution', metadata.institutionName || '—'],
      ['Series', String(metadata.series.length)],
      ['Marked regions', String(findingsCount)],
      ['Generated by', providerLabel],
    ];
    const rowH = 9;
    const rows = Math.ceil(pairs.length / 2);
    const boxH = rows * rowH + 3;
    this.ensure(boxH + 2);
    this.fill(PANEL);
    this.stroke(LINE);
    this.doc.setLineWidth(0.3);
    this.doc.rect(MARGIN, this.y, CONTENT_W, boxH, 'FD');
    const colW = CONTENT_W / 2;
    let yy = this.y + 5;
    for (let i = 0; i < pairs.length; i += 2) {
      this.pair(pairs[i], MARGIN + 3, yy, colW - 6);
      if (pairs[i + 1]) this.pair(pairs[i + 1], MARGIN + colW + 1, yy, colW - 6);
      yy += rowH;
    }
    this.y += boxH + 3;
  }

  private pair([label, value]: [string, string], x: number, yy: number, w: number) {
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(7);
    this.text(MUTED);
    this.doc.text(label.toUpperCase(), x, yy);
    this.doc.setFont('helvetica', 'normal');
    this.doc.setFontSize(9);
    this.text(INK);
    const v = this.doc.splitTextToSize(value || '—', w)[0] ?? '—';
    this.doc.text(v, x, yy + 4);
  }

  section(title: string) {
    this.ensure(12);
    this.y += 3;
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(11);
    this.text(ACCENT);
    this.doc.text(title.toUpperCase(), MARGIN, this.y);
    this.y += 2;
    this.stroke(LINE);
    this.doc.setLineWidth(0.3);
    this.doc.line(MARGIN, this.y, MARGIN + CONTENT_W, this.y);
    this.y += 4.5;
  }

  subheading(title: string) {
    this.ensure(8);
    this.y += 1.5;
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(9.5);
    this.text(INK);
    this.doc.text(title, MARGIN, this.y);
    this.y += 4.5;
  }

  paragraph(text: string, opts: ParagraphOpts = {}) {
    const { size = 10, color = INK, style = 'normal', gapAfter = 2.5, indent = 0 } = opts;
    if (!text || !text.trim()) return;
    this.doc.setFont('helvetica', style);
    this.doc.setFontSize(size);
    this.text(color);
    const lh = size * 0.42;
    const lines = this.doc.splitTextToSize(text.trim(), CONTENT_W - indent);
    for (const ln of lines) {
      this.ensure(lh);
      this.doc.text(ln, MARGIN + indent, this.y);
      this.y += lh;
    }
    this.y += gapAfter;
  }

  private pill(text: string, x: number, yTop: number, color: RGB) {
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(6.5);
    const w = this.doc.getTextWidth(text) + 3;
    const h = 3.8;
    this.fill(color);
    this.doc.roundedRect(x, yTop, w, h, 0.8, 0.8, 'F');
    this.text(WHITE);
    this.doc.text(text, x + 1.5, yTop + 2.7);
  }

  findingBlock(f: ExportFinding, index: number, img?: AnnotatedSliceImage) {
    this.ensure(10);
    this.y += 1;
    // Wrap the title (free-text, can be long); reserve a gutter for the tier pill
    // so it always lands beside the last line rather than off the right edge.
    const titleText = `${index}. ${f.title}`;
    const pillGutter = f.tier ? 26 : 0;
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(10.5);
    this.text(INK);
    const titleLines = this.doc.splitTextToSize(titleText, CONTENT_W - pillGutter);
    for (let li = 0; li < titleLines.length; li++) {
      this.ensure(5);
      this.doc.setFont('helvetica', 'bold');
      this.doc.setFontSize(10.5);
      this.text(INK);
      this.doc.text(titleLines[li], MARGIN, this.y);
      if (f.tier && li === titleLines.length - 1) {
        const w = this.doc.getTextWidth(titleLines[li]);
        this.pill(f.tier.toUpperCase(), MARGIN + w + 3, this.y - 3.1, tierColor(f.tier));
      }
      this.y += 4.8;
    }

    if (f.seriesNumber || f.instanceNumber != null) {
      const loc = [
        f.seriesNumber ? `Series #${f.seriesNumber}` : '',
        f.instanceNumber != null ? `Slice ${f.instanceNumber}` : '',
      ]
        .filter(Boolean)
        .join('  ·  ');
      this.doc.setFont('helvetica', 'normal');
      this.doc.setFontSize(8);
      this.text(MUTED);
      this.ensure(4);
      this.doc.text(loc, MARGIN, this.y);
      this.y += 4;
    }

    this.paragraph(f.description, { size: 9.5, gapAfter: 2 });
    if (img) this.image(img);
    this.y += 2.5;
  }

  image(img: AnnotatedSliceImage, caption?: string) {
    const maxW = 92;
    const maxH = 92;
    const ar = img.height / img.width || 1;
    let w = maxW;
    let h = w * ar;
    if (h > maxH) { h = maxH; w = h / ar; }
    this.ensure(h + (caption ? 6 : 2));
    this.doc.addImage(img.dataUrl, 'JPEG', MARGIN, this.y, w, h, undefined, 'FAST');
    this.stroke(LINE);
    this.doc.setLineWidth(0.2);
    this.doc.rect(MARGIN, this.y, w, h);
    this.y += h + 2.5;
    if (caption) {
      this.doc.setFont('helvetica', 'italic');
      this.doc.setFontSize(7.5);
      this.text(MUTED);
      const c = this.doc.splitTextToSize(caption, CONTENT_W);
      this.doc.text(c, MARGIN, this.y);
      this.y += c.length * 3.4 + 1.5;
    }
  }

  transcriptEntry(msg: ChatMessage) {
    const role = msg.role === 'user' ? 'Clinician' : 'AI analysis';
    this.ensure(7);
    this.y += 1.5;
    this.doc.setFont('helvetica', 'bold');
    this.doc.setFontSize(8.5);
    this.text(msg.role === 'user' ? ACCENT : INK);
    this.doc.text(role, MARGIN, this.y);
    this.y += 4;
    this.paragraph(stripMarkdown(msg.content), { size: 9, color: INK, gapAfter: 2 });
  }

  /** Draw the footer (with final page numbers) on every page. Call last. */
  finishFooters() {
    const total = this.doc.getNumberOfPages();
    for (let p = 1; p <= total; p++) {
      this.doc.setPage(p);
      this.stroke(LINE);
      this.doc.setLineWidth(0.2);
      this.doc.line(MARGIN, FOOTER_Y - 3.5, MARGIN + CONTENT_W, FOOTER_Y - 3.5);
      this.doc.setFont('helvetica', 'normal');
      this.doc.setFontSize(7);
      this.text(MUTED);
      this.doc.text('DICOMassist · Research/educational tool — not for clinical use', MARGIN, FOOTER_Y);
      this.doc.text(`Page ${p} of ${total}`, MARGIN + CONTENT_W, FOOTER_Y, { align: 'right' });
    }
  }
}
