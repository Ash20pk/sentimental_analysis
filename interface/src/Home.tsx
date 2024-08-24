import React, { useState } from 'react';
import { Container, Typography, Box, CircularProgress } from '@mui/material';
import axios from 'axios';
import SentimentAnalysisForm from './nillion/components/SentimentalAnalysisForm';

export default function Home() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (text: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:8000/analyze_sentiment', { text });
      setResult(response.data);
    } catch (err) {
      setError('An error occurred while analyzing the sentiment. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'green';
      case 'negative':
        return 'red';
      default:
        return 'grey';
    }
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h4" component="h1" gutterBottom>
        Blind Inference Demo for Sentiment Analysis
      </Typography>
      <Typography variant="body1" paragraph>
        Enter text to analyze its sentiment. The inference is "blind" because the party running the 
        sentiment analysis never sees the trained model state, which is provided to the Nada program as a secret.
      </Typography>

      <SentimentAnalysisForm onSubmit={handleAnalyze} isLoading={loading} />

      {loading && <CircularProgress />}

      {error && (
        <Typography color="error" variant="body1">
          {error}
        </Typography>
      )}

      {result && (
        <Box mt={4}>
          <Typography variant="h5" gutterBottom>
            Sentiment Analysis Result:
          </Typography>
          <Typography variant="body1" style={{ color: getSentimentColor(result.sentiment) }}>
            Sentiment: {result.sentiment}
          </Typography>
          <Typography variant="body1">
            Probability: {(result.probability * 100).toFixed(2)}%
          </Typography>
          <Typography variant="body1">
            Confidence: {result.confidence.toFixed(2)}%
          </Typography>
          <Typography variant="body2" mt={2}>
            {result.sentiment === "neutral" 
              ? "The model is uncertain about this prediction."
              : result.sentiment === "positive"
                ? `The model is ${result.confidence.toFixed(2)}% confident that the sentiment is positive.`
                : `The model is ${result.confidence.toFixed(2)}% confident that the sentiment is negative.`
            }
          </Typography>
        </Box>
      )}
    </Container>
  );
}