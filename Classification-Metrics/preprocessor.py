from nltk.tokenize import sent_tokenize, word_tokenize
import re

from regex import regex

# Large piece of AI-generated text from Chat-GPT (Prompt : Write me a story about space)
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

# Test input text
test_text = "Hello there, my name is Ciarán. I am training to be a software engineer."

# Excerpt from Chapter 2, Harry Potter and the Philosopher's Stone
text_harry_potter = """Nearly ten years had passed since the Dursleys had woken up to find their nephew on the front step, 
but Privet Drive had hardly changed at all. The sun rose on the same tidy front gardens and lit up the brass number 
four on the Dursleys' front door; it crept into their living room, which was almost exactly the same as it had been 
on the night when Mr. Dursley had seen that fateful news report about the owls. Only the photographs on the 
mantelpiece really showed how much time had passed. Ten years ago, there had been lots of pictures of what looked 
like a large pink beach ball wearing different-colored bonnets - but Dudley Dursley was no longer a baby, and now the 
photographs showed a large blond boy riding his first bicycle, on a carousel at the fair, playing a computer game 
with his father, being hugged and kissed by his mother. The room held no sign at all that another boy lived in the 
house, too.

Yet Harry Potter was still there, asleep at the moment, but not for long. His Aunt Petunia was awake and it was her 
shrill voice that made the first noise of the day.

"Up! Get up! Now!"

Harry woke with a start. His aunt rapped on the door again.

"Up!" she screeched. Harry heard her walking toward the kitchen and then the sound of the frying pan being put on the 
stove. He rolled onto his back and tried to remember the dream he had been having. It had been a good one. There had 
been a flying motorcycle in it. He had a funny feeling he'd had the same dream before.

His aunt was back outside the door.

"Are you up yet?" she demanded.

"Nearly," said Harry.

"Well, get a move on, I want you to look after the bacon. And don't you dare let it burn, I want everything perfect 
on Duddy's birthday."

Harry groaned.

"What did you say?" his aunt snapped through the door.

"Nothing, nothing . . ."

Dudley's birthday - how could he have forgotten? Harry got slowly out of bed and started looking for socks. He found 
a pair under his bed and, after pulling a spider off one of them, put them on. Harry was used to spiders, because the 
cupboard under the stairs was full of them, and that was where he slept.

When he was dressed he went down the hall into the kitchen. The table was almost hidden beneath all Dudley's birthday 
presents. It looked as though Dudley had gotten the new computer he wanted, not to mention the second television and 
the racing bike. Exactly why Dudley wanted a racing bike was a mystery to Harry, as Dudley was very fat and hated 
exercise - unless of course it involved punching somebody. Dudley's favorite punching bag was Harry, but he couldn't 
often catch him. Harry didn't look it, but he was very fast.

Perhaps it had something to do with living in a dark cupboard, but Harry had always been small and skinny for his 
age. He looked even smaller and skinnier than he really was because all he had to wear were old clothes of Dudley's, 
and Dudley was about four times bigger than he was. Harry had a thin face, knobbly knees, black hair, and bright 
green eyes. He wore round glasses held together with a lot of Scotch tape because of all the times Dudley had punched 
him on the nose. The only thing Harry liked about his own appearance was a very thin scar on his forehead that was 
shaped like a bolt of lightning. He had had it as long as he could remember, and the first question he could ever 
remember asking his Aunt Petunia was how he had gotten it.

"In the car crash when your parents died," she had said. "And don't ask questions."

Don't ask questions - that was the first rule for a quiet life with the Dursleys.

Uncle Vernon entered the kitchen as Harry was turning over the bacon.

"Comb your hair!" he barked, by way of a morning greeting.
"""


def split_text_into_sentences(input_text):
    sentences = sent_tokenize(input_text, language='English')
    # Ensures that each tokenized sentence contains at least one valid word
    filtered_sentences = [sentence for sentence in sentences if re.search(r'\w', sentence)]
    return filtered_sentences


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


def calculate_num_of_characters(input_text):
    """Calculates the number of characters in an input text"""
    return len(input_text)


def remove_new_line_characters(input_text):
    processed_text = input_text.replace('\n', '')
    return processed_text


# Ad Hoc Function Testing
# print("Number of characters :", calculate_num_of_characters(text_harry_potter))
# print("Number of words :", calculate_number_of_words(text_harry_potter))
#
# sentences = split_text_into_sentences(text_harry_potter)
# display_text_sentence_stats(sentences)
