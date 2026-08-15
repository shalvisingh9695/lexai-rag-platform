/**
 * Text Chunking Utility for RAG
 * Splits page-wise extracted text into meaningful chunks with overlapping context.
 */

/**
 * Page-aware Document Chunker for RAG.
 * Takes page-wise text array [{ page: 1, text: "..." }] and chunks each page
 * preserving paragraph/clause structure, word boundaries, and attached page numbers.
 *
 * @param {Array<{page: number, text: string}>} pages 
 * @param {string} documentId 
 * @param {string} filename 
 * @param {object} options 
 * @returns {Array<{chunkId: string, page: number, text: string, documentId: string, filename: string}>}
 */
export function chunkDocumentPages(pages, documentId = 'doc_1', filename = 'document.pdf', options = {}) {
  const chunkSize = options.chunkSize || 250; // ~250 words per chunk (~1200-1500 chars)
  const overlap = options.overlap || 40;       // ~40 words overlap (~200 chars)

  if (!Array.isArray(pages) || pages.length === 0) {
    return [];
  }

  const chunks = [];

  for (const pageObj of pages) {
    const pageNum = pageObj.page || 1;
    const rawText = pageObj.text || '';

    if (!rawText.trim()) continue;

    // Split page text into paragraphs/sections to preserve logical clause boundaries
    const paragraphs = rawText
      .split(/\n\s*\n/)
      .map(p => p.replace(/[ \t]+/g, ' ').trim())
      .filter(p => p.length > 0);

    let currentWords = [];
    let chunkIndex = 1;

    for (const para of paragraphs) {
      const paraWords = para.split(' ');

      // If adding this paragraph exceeds chunkSize and we already have accumulated words
      if (currentWords.length > 0 && (currentWords.length + paraWords.length) > chunkSize) {
        const chunkText = currentWords.join(' ');
        const chunkId = `chunk_${documentId}_p${pageNum}_${chunkIndex}`;

        chunks.push({
          chunkId,
          page: pageNum,
          text: chunkText,
          documentId,
          filename
        });

        chunkIndex++;

        // Keep last 'overlap' words for context continuity
        const overlapStart = Math.max(0, currentWords.length - overlap);
        currentWords = currentWords.slice(overlapStart);
      }

      // If a single paragraph itself exceeds chunkSize, split it by sliding window
      if (paraWords.length > chunkSize) {
        let pStart = 0;
        while (pStart < paraWords.length) {
          const pEnd = Math.min(pStart + chunkSize, paraWords.length);
          const sliceWords = paraWords.slice(pStart, pEnd);

          const combinedWords = pStart === 0 ? [...currentWords, ...sliceWords] : sliceWords;
          const chunkText = combinedWords.join(' ');
          const chunkId = `chunk_${documentId}_p${pageNum}_${chunkIndex}`;

          chunks.push({
            chunkId,
            page: pageNum,
            text: chunkText,
            documentId,
            filename
          });

          chunkIndex++;
          pStart += (chunkSize - overlap);
          currentWords = paraWords.slice(Math.max(0, pEnd - overlap));

          if (chunkSize <= overlap) break;
        }
      } else {
        currentWords.push(...paraWords);
      }
    }

    // Flush any remaining accumulated words for this page
    if (currentWords.length > 0) {
      const chunkText = currentWords.join(' ');
      const chunkId = `chunk_${documentId}_p${pageNum}_${chunkIndex}`;

      chunks.push({
        chunkId,
        page: pageNum,
        text: chunkText,
        documentId,
        filename
      });
    }
  }

  return chunks;
}

/**
 * Basic word-based chunker fallback for plain strings
 */
export function chunkText(text, options = {}) {
  const chunkSize = options.chunkSize || 250;
  const overlap = options.overlap || 40;

  if (!text || typeof text !== 'string') {
    return [];
  }

  const cleanText = text.replace(/\s+/g, ' ').trim();
  if (!cleanText) return [];

  const pages = [{ page: 1, text: cleanText }];
  return chunkDocumentPages(pages, options.documentId || 'doc_1', options.filename || 'document.pdf', { chunkSize, overlap });
}
