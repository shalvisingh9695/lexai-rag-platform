import { GoogleGenAI } from '@google/genai';
import { searchChunksForQuery } from './searchController.js';

/**
 * Grounded RAG Answer Endpoint Handler
 * POST /api/ask
 * 
 * Takes a user question:
 * 1. Generates query embedding & retrieves Top-K document chunks via vector cosine similarity.
 * 2. Formats strict context prompt grounding the Gemini model to provided excerpts ONLY.
 * 3. Calls Gemini API to generate answer grounded strictly in retrieved context.
 * 4. Returns answer + sources (without exposing raw vector embeddings).
 */
export async function handleAskQuestion(req, res) {
  try {
    const { query, topK, documentId } = req.body || {};

    // 1. Error handling: Empty query check
    if (!query || typeof query !== 'string' || !query.trim()) {
      return res.status(400).json({
        success: false,
        error: 'INVALID_QUERY: "query" text string is required in request body.'
      });
    }

    const userQuery = query.trim();
    const k = Math.min(Math.max(parseInt(topK, 10) || 3, 1), 5); // limit top 3-5 chunks

    // 2. Retrieve Top-K chunks using vector similarity
    let topChunks = [];
    try {
      topChunks = await searchChunksForQuery(userQuery, { topK: k, documentId });
    } catch (searchErr) {
      console.error('Vector search failed in RAG flow:', searchErr.message);
      return res.status(500).json({
        success: false,
        error: `VECTOR_SEARCH_FAILED: Could not retrieve relevant chunks. ${searchErr.message}`
      });
    }

    // Handle case where no relevant document chunks exist in database
    if (topChunks.length === 0) {
      return res.status(200).json({
        success: true,
        answer: 'The document does not contain sufficient information to answer this question.',
        sources: []
      });
    }

    // 3. Combine query + retrieved chunks into context prompt
    const topKChunksText = topChunks
      .map((c, i) => `[Excerpt ${i + 1}] (Chunk ID: ${c.chunkId}, Page ${c.page}, File: ${c.filename}):\n${c.text}`)
      .join('\n\n');

    const prompt = `You are a legal document assistant.

Answer the question ONLY using the provided document excerpts.

If the answer is not clearly present, say that the document does not contain enough information.

Question:
${userQuery}

Context:
${topKChunksText}

Answer:`;

    // 4. Send to Gemini model
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
      return res.status(500).json({
        success: false,
        error: 'GEMINI_API_FAILURE: GEMINI_API_KEY environment variable is missing or unconfigured.'
      });
    }

    let answerText = '';
    try {
      const ai = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: { 'User-Agent': 'aistudio-build' }
        }
      });

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: prompt
      });

      if (response && response.text) {
        answerText = response.text.trim();
      } else {
        answerText = 'The document does not contain sufficient information to answer this question.';
      }
    } catch (geminiErr) {
      console.error('Gemini RAG answer generation failed:', geminiErr.message);
      return res.status(500).json({
        success: false,
        error: `GEMINI_API_FAILURE: Failed to generate answer from Gemini model. ${geminiErr.message || geminiErr}`
      });
    }

    // 5. Clean sources payload (no embeddings sent)
    const sources = topChunks.map(c => ({
      chunkId: c.chunkId,
      page: c.page,
      text: c.text,
      similarity: c.similarity
    }));

    return res.status(200).json({
      success: true,
      answer: answerText,
      sources
    });
  } catch (err) {
    console.error('Error during RAG ask handler:', err);
    return res.status(500).json({
      success: false,
      error: `RAG_FAILED: ${err.message || 'An unexpected error occurred during RAG answer generation.'}`
    });
  }
}
