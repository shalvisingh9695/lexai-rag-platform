import Document from '../models/Document.js';
import { getInMemoryStore, isUsingMemory } from '../utils/dbConnect.js';
import { generateEmbedding } from '../rag/embed.js';
import { cosineSimilarity } from '../rag/search.js';

/**
 * Perform vector search over stored document chunks for a given query text.
 * Returns top-K matching chunks with similarity scores.
 */
export async function searchChunksForQuery(query, options = {}) {
  const { topK = 3, documentId } = options;

  if (!query || typeof query !== 'string' || !query.trim()) {
    return [];
  }

  // 1. Convert user query -> embedding vector (gemini-embedding-2-preview)
  const queryEmbedding = await generateEmbedding(query.trim());

  // 2. Fetch all document chunks from MongoDB or in-memory fallback store
  let allChunks = [];

  if (isUsingMemory()) {
    const store = getInMemoryStore();
    const docs = store.documents || [];
    for (const doc of docs) {
      if (documentId && String(doc._id) !== String(documentId) && String(doc.id) !== String(documentId)) {
        continue;
      }
      if (Array.isArray(doc.chunks)) {
        for (const chunk of doc.chunks) {
          allChunks.push({
            chunkId: chunk.chunkId,
            documentId: chunk.documentId || doc._id,
            page: chunk.page,
            text: chunk.text,
            filename: chunk.filename || doc.fileName || 'document.pdf',
            embedding: chunk.embedding
          });
        }
      }
    }
  } else {
    const filter = documentId ? { _id: documentId } : {};
    const docs = await Document.find(filter, { chunks: 1, fileName: 1, _id: 1 }).lean();
    for (const doc of docs) {
      if (Array.isArray(doc.chunks)) {
        for (const chunk of doc.chunks) {
          allChunks.push({
            chunkId: chunk.chunkId,
            documentId: chunk.documentId || doc._id.toString(),
            page: chunk.page,
            text: chunk.text,
            filename: chunk.filename || doc.fileName || 'document.pdf',
            embedding: chunk.embedding
          });
        }
      }
    }
  }

  if (allChunks.length === 0) {
    return [];
  }

  // 3. Compute cosine similarity between query embedding and each chunk embedding
  const scoredChunks = [];

  for (const chunk of allChunks) {
    if (!Array.isArray(chunk.embedding) || chunk.embedding.length === 0) {
      continue;
    }

    const score = cosineSimilarity(queryEmbedding, chunk.embedding);
    
    scoredChunks.push({
      chunkId: chunk.chunkId,
      documentId: chunk.documentId,
      page: chunk.page || 1,
      text: chunk.text,
      similarity: parseFloat(score.toFixed(4)),
      filename: chunk.filename
    });
  }

  // 4. Sort by similarity score in descending order (highest first)
  scoredChunks.sort((a, b) => b.similarity - a.similarity);

  // 5. Select Top-K results
  return scoredChunks.slice(0, topK);
}

/**
 * Semantic Vector Search Endpoint Handler
 * POST /api/search
 */
export async function handleSemanticSearch(req, res) {
  try {
    const { query, topK, documentId } = req.body || {};

    // Error Handling: Empty query check
    if (!query || typeof query !== 'string' || !query.trim()) {
      return res.status(400).json({
        success: false,
        error: 'INVALID_QUERY: "query" text string is required in request body.'
      });
    }

    const k = parseInt(topK, 10) || 3;

    let topResults = [];
    try {
      topResults = await searchChunksForQuery(query, { topK: k, documentId });
    } catch (embedErr) {
      console.error('Failed to generate query embedding or search chunks:', embedErr.message);
      return res.status(500).json({
        success: false,
        error: `EMBEDDING_FAILED: Could not perform search. ${embedErr.message}`
      });
    }

    return res.status(200).json({
      success: true,
      results: topResults
    });
  } catch (err) {
    console.error('Error during vector search:', err);
    return res.status(500).json({
      success: false,
      error: `SEARCH_FAILED: ${err.message || 'An unexpected error occurred during vector search.'}`
    });
  }
}
