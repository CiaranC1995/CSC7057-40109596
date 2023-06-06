import unittest
from src.ClassificationMetrics.TextPreprocessor import preprocessor as pp


class TestCharacterCount(unittest.TestCase):

    def test_valid_input(self):
        # Valid input: Check if the character count is calculated correctly
        input_text = "Hello, World!"
        expected_output = 13
        actual_count = pp.TextPreprocessor.num_of_chars(input_text)
        self.assertEqual(actual_count, expected_output)

    def test_all_numbers(self):
        # Valid input: Check if the character count is calculated correctly
        input_text = "123456789"
        expected_output = 9
        actual_count = pp.TextPreprocessor.num_of_chars(input_text)
        self.assertEqual(actual_count, expected_output)

    def test_empty_input(self):
        # Invalid input: Check if the function handles empty input correctly
        input_text = ""
        expected_output = 0
        actual_count = pp.TextPreprocessor.num_of_chars(input_text)
        self.assertEqual(actual_count, expected_output)

    def test_unicode_input(self):
        # Valid input with Unicode characters: Check if the function handles Unicode correctly
        input_text = "Héllò, 世界!"
        expected_output = 10
        actual_count = pp.TextPreprocessor.num_of_chars(input_text)
        self.assertEqual(actual_count, expected_output)

    def test_no_input(self):
        # Invalid Input: Check if the function throws a TypeError when it is called with no input
        input_text = None
        self.assertRaises(TypeError, pp.TextPreprocessor.split_into_sentences, input_text)


if __name__ == '__main__':
    unittest.main()
