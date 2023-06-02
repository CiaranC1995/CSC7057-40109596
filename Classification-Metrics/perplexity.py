import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def calculate_perplexity(text, model, tokenizer):
    """Calculates the perplexity of a given input text."""
    inputs = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()


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

text_human = "This is a sample sentence written by a human."

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

perplexity_ai = calculate_perplexity(text_ai, model, tokenizer)
perplexity_human = calculate_perplexity(text_human, model, tokenizer)

print(f"Perplexity (AI-generated text): {perplexity_ai}")
print(f"Perplexity (Human-generated text): {perplexity_human}")
