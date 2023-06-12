from src.ClassificationMetrics import perplexity, sentenceAnalysis, sentimentAnalysis, vocab_richness

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\Human\Human_HarryPotter.txt') as f:
    human_text_to_analyse = f.read()

# Reading in text to analyse from file
with open(r'C:\Users\ccase\Desktop\Dissertation\Examples of Text\AI\AI_SpaceStory.txt') as f:
    ai_text_to_analyse = f.read()


class CalcFeatures:

    def __init__(self):
        pass

    @staticmethod
    def evaluate_text_features(input_text):
        """Evaluates numeric values for each of the following features; perplexity, burstiness, sentiment analysis,
        Flesch-Kincaid readability score, Coleman-Liau readability index, average sentence length and vocabulary richness
        in that exact order."""
        feature_scores = []

        # Perplexity and Burstiness
        ppl_burstiness = perplexity.PerplexityBurstiness.process_text_ppl_burstiness(input_text)
        feature_scores.append(ppl_burstiness['avg_text_ppl'])
        feature_scores.append(ppl_burstiness['text_burstiness'])

        # Sentiment Analysis
        sentiment_score = sentimentAnalysis.SentimentAnalysis.perform_average_sentiment_analysis(input_text)
        feature_scores.append(sentiment_score)

        # Flesch-Kincaid Readability Score
        fk_score = sentenceAnalysis.SentenceAnalysis.calculate_flesch_kincaid(input_text)
        feature_scores.append(fk_score)

        # Coleman-Liau Readability Score
        cl_score = sentenceAnalysis.SentenceAnalysis.calculate_coleman_liau(input_text)
        feature_scores.append(cl_score)

        # Average Sentence Length
        avg_sentence_length = sentenceAnalysis.SentenceAnalysis.average_sentence_length_words(input_text)
        feature_scores.append(avg_sentence_length)

        # Vocabulary Richness
        v_rich = vocab_richness.VocabRichness.calculate_total_vocab_richness_no_stopwords(input_text)
        feature_scores.append(v_rich)

        return feature_scores


print(CalcFeatures.evaluate_text_features(human_text_to_analyse))
