from nltk.tokenize import sent_tokenize, word_tokenize
import re

from regex import regex

input_text = """Once upon a time, in a vast and wondrous universe, there existed a small blue planet called Earth. On 
this planet, humans marveled at the twinkling stars that adorned the night sky and dreamed of exploring the mysteries 
of space.

In a modest suburban neighborhood, young Emma gazed at the stars every night from her backyard. She was captivated by 
the endless possibilities that lay beyond Earth's atmosphere. Emma's room was filled with books about galaxies, 
spaceships, and the wonders of the universe. Her imagination soared as she envisioned herself floating among the stars.

One day, as Emma was reading about the latest discoveries in space, she came across an advertisement for a unique 
opportunity—a chance to participate in an intergalactic adventure. It was a competition organized by a renowned space 
agency, seeking a young explorer to join their crew and embark on a journey to explore uncharted territories in the 
far reaches of the universe.

Emma's heart raced with excitement. She knew this was her chance to fulfill her lifelong dream of venturing into 
space. With determination, she submitted an application and poured her heart into the accompanying essay, describing 
her unwavering passion for the cosmos.

Weeks passed, and Emma's anticipation grew. Then, one fateful afternoon, she received a letter bearing the space 
agency's emblem. Trembling with excitement, she tore open the envelope, her eyes scanning the words: 
"Congratulations, Emma! You have been selected as the young explorer for our intergalactic mission."

Emma could hardly contain her joy. She hugged her parents tightly, their eyes filled with pride. She spent the 
following weeks undergoing rigorous training, preparing her mind and body for the extraordinary challenges that 
awaited her in the great expanse of space.

Finally, the day of departure arrived. Emma, now suited up in her astronaut gear, stepped into the spacecraft. She 
looked out the window and bid farewell to her family, knowing that she was about to embark on a journey that would 
change her life forever.

As the spacecraft soared into the endless depths of space, Emma marveled at the celestial wonders that unfolded 
before her eyes. She witnessed nebula's bursting with vibrant colors, swirling galaxies stretching across the cosmos, 
and the breathtaking beauty of distant planets.

During her mission, Emma and her fellow astronauts encountered intelligent alien civilizations, each with its unique 
customs and ways of life. They shared knowledge, experiences, and formed deep bonds across the vastness of space.

Years passed, and Emma became an inspiration to aspiring young explorers back on Earth. Her remarkable journey and 
discoveries opened the doors to new possibilities for humanity, igniting a passion for space exploration among 
generations to come.

Finally, it was time for Emma to return home. As the spacecraft re-entered Earth's atmosphere, she experienced a mix 
of emotions—gratitude, awe, and a sense of accomplishment. She had become part of something far greater than herself, 
leaving an indelible mark on the history of human exploration.

Emma's feet touched the familiar ground of Earth once again, but she was forever transformed by her journey through 
the stars. She knew that the wonders of space would continue to beckon future generations, inspiring them to reach 
for the stars, just as she had done.

And so, the tale of Emma, the young explorer who traversed the cosmos, would forever serve as a reminder of the 
boundless human spirit and our insatiable desire to unravel the mysteries of the universe."""

test_text = "Hello there, my name is Ciarán. I am training to be a software engineer."


def split_text_into_sentences(input_text):
    sentences = sent_tokenize(input_text, language='English')
    return sentences


def display_text_sentence_stats(sentences):
    """Displays some stats on the sentences within a text ******************** not finished yet"""
    print('Number of Sentences :', len(sentences))

    total_length_of_sentences = 0
    for sent in sentences:
        total_length_of_sentences += len(sent)
    print('Average Number of Sentence Characters :', total_length_of_sentences / len(sentences))


def separate_words_with_hyphen(text):
    """*********************************** needs some work is not working properly yet"""
    pattern = r'\b(?<!\w+-)(\w+)([-—])(?!\w+-)(\w+)\b'
    separated_text = regex.sub(pattern, r'\1 \2 \3', text)
    return separated_text


def calculate_number_of_words(input_text):
    """Calculates the number of words in an input text"""
    tokens = word_tokenize(separate_words_with_hyphen(input_text))
    filtered_tokens = [token for token in tokens if re.match(r'^[a-zA-ZÀ-ÿ0-9]+$', token)]
    word_count = len(filtered_tokens)
    return word_count


count = calculate_number_of_words(input_text)
print(count)

#print(separate_words_with_hyphen(input_text))
