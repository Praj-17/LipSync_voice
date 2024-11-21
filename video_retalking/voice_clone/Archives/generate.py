from transformers import AutoProcessor, BarkModel

import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to("cuda")

def generate_audio(text, preset, output):
    inputs = processor(text, voice_preset = preset)
    for k, v in inputs.items():
        inputs[k] = v.to("cuda")
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output, rate = sample_rate, data =audio_array)

text = """
The Curse of the Black Horizon

The salty sea breeze swept across the deck of the Sable Serpent, a ship as infamous as the crew that sailed her. With sails as black as midnight and a flag that bore a crimson serpent coiled around a skull, the ship cut through the waves like a predator on the hunt. Captain Elias "Iron Hand" Vane stood at the helm, his weathered face half-lit by the setting sun. His left hand, replaced by a glinting steel hook after a skirmish with the Royal Navy, clinked against the wheel as he turned the ship towards their next destination: the fabled Black Horizon.

Legends of the Black Horizon had long been whispered in seafarers' taverns and pirate dens. It was said to be a cursed stretch of sea where the sun never rose, shrouded in eternal twilight. At its heart lay the Obsidian Vault, an island said to hold treasures beyond imagination but guarded by horrors that turned men mad. Many ships had ventured there, but none had returned.

Elias had heard the stories since he was a boy. Unlike others who were content to let the legend remain a mystery, he was determined to claim its riches. The map to the Black Horizon, tattooed on the back of a traitorous quartermaster Elias had once marooned, had cost him dearly—three ships and half his crew. Now, with the map seared into his mind and a handpicked crew of the most ruthless pirates, he was ready.

The crew buzzed with a mix of excitement and fear as the Sable Serpent approached the edge of the Black Horizon. The sea grew unnaturally calm, the waves flattening into an oily black mirror that reflected the bruised-purple sky. The air grew cold, and an eerie silence replaced the constant cries of gulls and the crash of waves. It was as if the world itself held its breath.

"Cap'n, I don’t like this," muttered Finn O’Malley, the ship's superstitious bosun. He clutched a rosary he’d stolen from a missionary years ago. "The stories—"

"Are stories," Elias interrupted, his voice sharp. "There’s treasure ahead, O’Malley. Enough to make us all kings."

"Or corpses," muttered O’Malley under his breath, but he fell silent when Elias shot him a glare.

As the ship ventured further into the twilight zone, the crew began to notice strange things. Shadows danced on the deck with no apparent source. Whispers echoed in the still air, speaking in languages no one could understand. Tobias "Two-Eye" McGraw, the ship’s sharpest lookout, swore he saw figures moving beneath the water, their eyes glowing faintly.

By the second day, the unease had turned into dread. The crew grew restless, and mutinous whispers spread like wildfire. Elias knew he had to act fast. That night, as the ship anchored in a sea that seemed alive, he gathered the crew on deck.

"Listen well," he growled, his hook glinting in the dim lantern light. "We’ve come too far to turn back now. The Black Horizon tests the weak, but we are stronger than any sailor that’s come before. Think of the gold, the jewels, the power waiting for us. Do you want to return empty-handed, mocked by every drunkard in Tortuga?"

The crew murmured their agreement, though uneasily. Elias smiled grimly, knowing fear was a leash he could use.
"""

text_2 = """
On the third day, they sighted the Obsidian Vault. It rose from the water like a jagged tooth, its cliffs black and glistening as if carved from volcanic glass. The island pulsed faintly with an unnatural light, and a thick mist clung to its shores.

Elias ordered the crew to prepare the rowboats. Only twenty would go ashore, the rest staying behind to guard the Sable Serpent. Finn O’Malley, Tobias McGraw, and Elias’s loyal first mate, Marlowe "Blackheart" Kane, were among those chosen. Armed with cutlasses, pistols, and lanterns, the party set off.

The island was deathly quiet. The sand was cold and black, crunching like glass beneath their boots. As they ventured inland, they found ruins—remnants of a once-great civilization. Pillars carved with serpentine patterns rose from the ground, their surfaces worn smooth by time. At the center of the island lay a vast temple, its entrance a gaping maw that exhaled a chill wind.

Inside, the air smelled of damp stone and decay. The walls were lined with murals depicting battles, sacrifices, and a serpent-headed deity whose eyes seemed to follow them. At the heart of the temple lay the treasure: piles of gold coins, jeweled crowns, and chests overflowing with pearls and rubies. But amidst the treasure stood a black obelisk, its surface swirling like liquid shadow.

Marlowe approached it cautiously. "What do you make of it, Cap’n?"

Elias stared at the obelisk, his greed warring with a gnawing sense of unease. "It’s power," he said finally. "Power beyond anything we’ve known."

As he reached out to touch it, Finn O’Malley grabbed his arm. "Don’t, Cap’n. Look around you—this place isn’t natural. We take the gold and leave the rest."

Elias shoved him aside. "Coward," he spat. "This is what we came for."

The moment his hook touched the obelisk, a deafening roar filled the temple. The murals on the walls came to life, the serpent deity writhing and hissing. Shadows poured from the obelisk, coalescing into humanoid forms with glowing eyes.

"Run!" Marlowe shouted, firing his pistol at the nearest shadow. The bullet passed through it harmlessly.

The crew scrambled to gather what treasure they could, but the shadows were relentless. They moved like smoke, their touch freezing flesh. Tobias McGraw fell first, his screams echoing as the shadows consumed him. Finn O’Malley dropped his rosary and fled, dragging a half-full chest of gold.

Elias, clutching a strange black gem he’d pried from the obelisk, led the survivors out of the temple and back to the boats. The island seemed alive now, the ground shaking and splitting as if trying to swallow them. They rowed frantically back to the Sable Serpent, the shadows pursuing them across the water.

By the time they reached the ship, only ten remained. The shadows did not follow them onto the Sable Serpent, but the survivors were changed. They spoke little, their eyes hollow.

Elias locked himself in his cabin, the black gem pulsing faintly on his desk. He stared at it, his mind filled with whispers of power and promises of immortality. But as the days passed, he began to see things—shadows moving in the corners of his vision, the faces of his dead crew in the waves.

The Sable Serpent never left the Black Horizon. Other ships that ventured near claimed to see her adrift, her black sails tattered and her deck empty. Some say Captain Elias still haunts the ship, cursed to guard his ill-gotten treasure for eternity.

And so the legend of the Black Horizon grew, a tale of greed, betrayal, and the price of ambition. Few dared to seek it, but those who did found only darkness—and death.
"""
generate_audio(text=text, preset="v2/en_speaker_9", output="output_2.wav")