import unittest
from src.ClassificationMetrics.TextPreprocessor import preprocessor as pp


class TestRemoveNewLineChars(unittest.TestCase):

    def test_remove_new_line_characters_valid_input(self):
        input_text = "There are multiple sentences in this example.\n With some new line \n characters too.\n"
        expected_output = "There are multiple sentences in this example. With some new line  characters too."
        processed_text = pp.TextPreprocessor.remove_new_line_chars(input_text)
        self.assertEqual(processed_text, expected_output)

    def test_remove_new_line_characters_empty_input(self):
        input_text = ""
        expected_output = ""
        processed_text = pp.TextPreprocessor.remove_new_line_chars(input_text)
        self.assertEqual(processed_text, expected_output)

    def test_remove_new_line_characters_no_newline_input(self):
        input_text = "There are no new line characters in this sentence."
        expected_output = "There are no new line characters in this sentence."
        processed_text = pp.TextPreprocessor.remove_new_line_chars(input_text)
        self.assertEqual(processed_text, expected_output)

    def test_remove_new_line_characters_invalid_input(self):
        input_text = None
        self.assertRaises(AttributeError, pp.TextPreprocessor.remove_new_line_chars, input_text)


if __name__ == '__main__':
    unittest.main()
