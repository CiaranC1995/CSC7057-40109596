import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from preprocessor import *

# Large piece of AI-generated text from Chat-GPT (Prompt : Write me a story about space)
text_ai = """Once upon a time, in a vast and wondrous universe, there existed a small blue planet called Earth. On 
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
before her eyes. She witnessed nebulas bursting with vibrant colors, swirling galaxies stretching across the cosmos, 
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

# Excerpt from Chapter 2, Harry Potter and the Philosopher's Stone
text_harry_potter_human = """Nearly ten years had passed since the Dursleys had woken up to find their nephew on the front step, 
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

# AI generated information on the principles of Agile software development
text_ai_software = """Agile software development is an iterative and flexible approach to software development that 
emphasizes collaboration, adaptability, and delivering value in short increments. It focuses on the following key 
principles:

Customer Satisfaction: Agile places a high priority on customer satisfaction by continuously delivering valuable 
software that meets customer needs. Regular customer involvement and feedback are encouraged throughout the 
development process.

Iterative and Incremental Development: Agile promotes breaking down the software development process into smaller 
iterations or timeboxed increments called sprints. Each sprint delivers a working product increment, allowing for 
continuous improvement and feedback.

Adaptive Planning: Agile acknowledges that requirements and priorities can change over time. It emphasizes 
adaptability by allowing for changes and adjustments to be incorporated throughout the project, ensuring that the 
development efforts align with the evolving needs of the project stakeholders.

Embracing Change: Agile recognizes change as an opportunity. It encourages embracing change requirements, even in the 
late stages of development, to provide the greatest value to customers. Changes are welcomed, and the development 
process is designed to be flexible enough to accommodate them.

Cross-functional Collaboration: Agile emphasizes close collaboration and effective communication between team 
members, including developers, testers, product owners, and customers. Regular interactions and face-to-face 
conversations are valued to foster better understanding, shared ownership, and collective decision-making.

Self-organizing Teams: Agile promotes the concept of self-organizing teams. Team members collaborate to determine how 
best to accomplish their work, make decisions collectively, and organize their tasks. This encourages empowerment, 
creativity, and collective accountability.

Continuous Improvement: Agile encourages a culture of continuous improvement. Teams regularly reflect on their work, 
identify areas for improvement, and make adjustments accordingly. Agile methodologies such as Scrum often include 
retrospective meetings at the end of each sprint to facilitate this reflection and learning.

Delivering Working Software: Agile prioritizes delivering working software as early and frequently as possible. Each 
iteration or increment focuses on delivering a potentially shippable product, ensuring that tangible value is 
produced at regular intervals.

Agile methodologies, such as Scrum, Kanban, and Extreme Programming (XP), embody these principles to varying degrees 
and provide specific frameworks and practices to guide the software development process.

Overall, agile software development promotes collaboration, flexibility, customer focus, and continuous improvement 
to maximize customer value and project success."""


def calculate_perplexity(text, model, tokenizer):
    """Calculates the perplexity of a given input text."""
    inputs = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

tokenized_text = split_text_into_sentences(text_ai_software)

# Remove new line characters from tokenized list of sentences
processed_tokenized_text = []
for sentence in tokenized_text:
    processed_sentence = remove_new_line_characters(sentence)
    processed_tokenized_text.append(processed_sentence)

# Calculate the individual PPL of each sentence
sentence_perplexities = []
for sentence in processed_tokenized_text:
    perplexity = calculate_perplexity(sentence, model, tokenizer)
    # print(ppl)
    sentence_perplexities.append(perplexity)

# Calculate average PPL of whole text
print(f'Average text Perplexity : {sum(sentence_perplexities) / len(sentence_perplexities)}')

# Print the sentence with the highest PPL
max_index = sentence_perplexities.index(max(sentence_perplexities))
print(
    f"Sentence with highest PPL of {round(max(sentence_perplexities), 2)} is sentence {max_index + 1} : '{processed_tokenized_text[max_index]}'")

# perplexity_ai = calculate_perplexity(text_ai, model, tokenizer)
# perplexity_human = calculate_perplexity(text_human, model, tokenizer)
#
# print(f"Perplexity (AI-generated text): {perplexity_ai}")
# print(f"Perplexity (Human-generated text): {perplexity_human}")
