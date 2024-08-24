import React, { useState } from 'react';
import {
  TextField,
  Button,
  Container,
} from '@mui/material';

interface SentimentAnalysisFormProps {
  onSubmit: (text: string) => void;
  isLoading: boolean;
}

const SentimentAnalysisForm: React.FC<SentimentAnalysisFormProps> = ({ onSubmit, isLoading }) => {
  const [text, setText] = useState('');

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setText(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    onSubmit(text);
  };

  return (
    <Container>
      <form onSubmit={handleSubmit}>
        <TextField
          id="sentiment-text"
          name="sentiment-text"
          label="Enter text for sentiment analysis"
          value={text}
          onChange={handleInputChange}
          required
          fullWidth
          margin="normal"
          multiline
          rows={4}
        />
        <Button 
          type="submit" 
          variant="contained" 
          color="primary"
          disabled={isLoading}
        >
          {isLoading ? 'Analyzing...' : 'Analyze Sentiment'}
        </Button>
      </form>
    </Container>
  );
};

export default SentimentAnalysisForm;