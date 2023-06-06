import unittest
from src.ClassificationMetrics.TextPreprocessor import preprocessor as pp


class TestRemovePunctuation(unittest.TestCase):

    def test_remove_punctuation_valid(self):
        # Valid Input : Test if punctuation is correctly removed
        input_text = "Hello, World! This is a sample text."
        expected_output = "Hello World This is a sample text"
        actual_output = pp.TextPreprocessor.remove_punctuation(input_text)
        self.assertEqual(actual_output, expected_output)

    def test_remove_punctuation_empty_text(self):
        # Valid Input : Test if function recognises empty string
        input_text = ""
        expected_output = ""
        actual_output = pp.TextPreprocessor.remove_punctuation(input_text)
        self.assertEqual(actual_output, expected_output)

    def test_remove_punctuation_no_punctuation(self):
        # Valid Input : Test if function recognises that there is no punctuation in input string
        input_text = "This text has no punctuation"
        expected_output = "This text has no punctuation"
        actual_output = pp.TextPreprocessor.remove_punctuation(input_text)
        self.assertEqual(actual_output, expected_output)

    def test_remove_punctuation_with_numbers(self):
        # Valid Input : Test if function can handle removal of punctuation in an input string containing numbers
        input_text = "Hello, 123! This is a sample text."
        expected_output = "Hello 123 This is a sample text"
        actual_output = pp.TextPreprocessor.remove_punctuation(input_text)
        self.assertEqual(actual_output, expected_output)

    def test_remove_punctuation_with_special_characters(self):
        # Valid Input : Test if function can remove special characters
        input_text = "Hello, @#$%!?<>+-/£^*() This text contains a lot of special characters."
        expected_output = "Hello  This text contains a lot of special characters"
        actual_output = pp.TextPreprocessor.remove_punctuation(input_text)
        self.assertEqual(actual_output, expected_output)

    def test_remove_punctuation_with_no_input(self):
        # Invalid Input : Test if function throws correct error when not given an input
        input_text = None
        self.assertRaises(TypeError, pp.TextPreprocessor.remove_punctuation, input_text)


if __name__ == '__main__':
    unittest.main()
