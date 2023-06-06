import unittest
from src.ClassificationMetrics.TextPreprocessor import preprocessor as pp


class TextProcessorSplitSentenceTest(unittest.TestCase):

    def test_split_into_sentences_valid_input_full_stops(self):
        input_text = "This is the first sentence. This is the second sentence. This is the third sentence."
        expected_output = [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence."
        ]
        processed_sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        self.assertEqual(processed_sentences, expected_output)

    def test_split_into_sentences_valid_input_exclamation(self):
        input_text = "This sentence ends with an exclamation mark! So does this one! And this one too!"
        expected_output = [
            "This sentence ends with an exclamation mark!",
            "So does this one!",
            "And this one too!"
        ]
        processed_sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        self.assertEqual(processed_sentences, expected_output)

    def test_split_into_sentences_valid_input_question(self):
        input_text = "This is a question? And again another question? Surely not, another question? This is hopefully " \
                     "the last question?"
        expected_output = [
            "This is a question?",
            "And again another question?",
            "Surely not, another question?",
            "This is hopefully the last question?"
        ]
        processed_sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        self.assertEqual(processed_sentences, expected_output)

    def test_split_into_sentences_valid_input_commas(self):
        input_text = "This sentence contains only commas, even at the end, it is not grammatically correct,"
        expected_output = [
            "This sentence contains only commas, even at the end, it is not grammatically correct,"
        ]
        processed_sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        self.assertEqual(processed_sentences, expected_output)

    def test_split_into_sentences_valid_input_special_chars(self):
        input_text = "How much will that cost? It will be $400, which is 25% off it's original value. Great, " \
                     "send the receipt to my address @ 34 Street Street."
        expected_output = [
            "How much will that cost?",
            "It will be $400, which is 25% off it's original value.",
            "Great, send the receipt to my address @ 34 Street Street."
        ]
        processed_sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        self.assertEqual(processed_sentences, expected_output)

    def test_split_into_sentences_empty_input(self):
        input_text = ""
        expected_output = []
        processed_sentences = pp.TextPreprocessor.split_into_sentences(input_text)
        self.assertEqual(processed_sentences, expected_output)

    def test_split_into_sentences_invalid_input(self):
        input_text = None
        self.assertRaises(TypeError, pp.TextPreprocessor.split_into_sentences, input_text)


if __name__ == '__main__':
    unittest.main()