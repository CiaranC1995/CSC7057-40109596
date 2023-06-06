import unittest


class TestCharacterCount(unittest.TestCase):
    @staticmethod
    def num_of_chars(input_text):
        """Calculates the number of characters in an input text"""
        return len(input_text)

    def test_valid_input(self):
        # Valid input: Check if the character count is calculated correctly
        input_text = "Hello, World!"
        expected_count = 13
        actual_count = self.num_of_chars(input_text)
        self.assertEqual(actual_count, expected_count)

    def test_empty_input(self):
        # Invalid input: Check if the function handles empty input correctly
        input_text = ""
        expected_count = 0
        actual_count = self.num_of_chars(input_text)
        self.assertEqual(actual_count, expected_count)

    def test_unicode_input(self):
        # Valid input with Unicode characters: Check if the function handles Unicode correctly
        input_text = "Héllò, 世界!"
        expected_count = 10
        actual_count = self.num_of_chars(input_text)
        self.assertEqual(actual_count, expected_count)


if __name__ == '__main__':
    unittest.main()
