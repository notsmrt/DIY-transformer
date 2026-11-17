attention_heads = 14
kv_heads = 2
head_dim = 64
n_layers = 24
hidden_size = 896
rope_theta = 1000000.0
rms_norm_eps = 1e-6
max_position_embeddings = 32768

# 1. Narrative: The Lighthouse
lighthouse_prompt = """
The lighthouse stood alone on the edge of the northern coastline, its white walls battered by decades of wind and salt. Thomas had been its keeper for forty-three years, and he knew every creaking floorboard and rusted hinge in the building.

Outside, the storm was growing worse. Waves crashed against the rocks below, sending white foam high into the darkness. Thomas adjusted the oil lamp and looked through the narrow window. Something unusual caught his attention.

Far beyond the reef, a ship was flashing a strange sequence of lights. Three short flashes, followed by two long ones. He reached for his navigation charts, but no vessel was marked in that direction.

"That's impossible," he whispered.

The radio crackled, producing only static. Another flash appeared, brighter than before. Thomas put on his heavy coat, grabbed his lantern, and opened the lighthouse door.

The wind nearly knocked him backward. He had spent his entire life warning ships away from these dangerous waters. Now, for the first time, he wondered whether someone was trying to guide him toward something instead.

He stepped into the storm, unaware that the ship had disappeared from the horizon.
"""

# 2. Code: Python Number Guessing Game
python_prompt = """
import random

# Generate a random number between 1 and 100.
secret_number = random.randint(1, 100)

attempts = 0
print("Welcome to the Number Guessing Game!")
print("I'm thinking of a number between 1 and 100.")

while True:
    try:
        guess = int(input("Enter your guess: "))
    except ValueError:
        print("Please enter a valid whole number.")
        continue

    if guess < 1 or guess > 100:
        print("Your guess must be between 1 and 100.")
        continue

    attempts += 1

    if guess < secret_number:
        print("Too low! Try again.")
    elif guess > secret_number:
        print("Too high! Try again.")
    else:
        print("Congratulations! You guessed the number!")
        print(f"It took you {attempts} attempts.")
        break

print("Thanks for playing!")
"""

# 3. Factual: Great Barrier Reef
reef_prompt = """
The Great Barrier Reef is the world's largest coral reef system, located in the Coral Sea off the northeastern coast of Queensland, Australia. It extends approximately 2,300 kilometres and consists of nearly 3,000 individual reefs and hundreds of islands. The ecosystem supports an extraordinary diversity of marine life, including more than 1,500 species of fish, hundreds of coral species, sea turtles, sharks, dolphins, and numerous seabirds.

Coral reefs develop through the accumulation of calcium carbonate skeletons produced by tiny marine organisms called coral polyps. Over long periods, these structures form complex habitats that provide food, shelter, and breeding grounds for many species.

Despite its ecological importance, the Great Barrier Reef faces significant environmental pressures. Rising ocean temperatures can cause coral bleaching, a process in which corals lose the microscopic algae that provide much of their energy. Ocean acidification, coastal pollution, agricultural runoff, and severe weather events also threaten reef health.

Conservation efforts include marine protected areas, water-quality improvement programs, restrictions on fishing, and scientific monitoring. International organizations and Australian authorities continue to study ways to improve the reef's resilience against climate change and other environmental threats.
"""

# 4. Recipe: Homemade Vegetable Soup
soup_prompt = """
Homemade Vegetable Soup

Ingredients:
- 2 tablespoons olive oil
- 1 medium onion, diced
- 2 carrots, peeled and sliced
- 2 celery stalks, chopped
- 2 potatoes, peeled and cubed
- 1 zucchini, chopped
- 2 cloves garlic, minced
- 6 cups vegetable broth
- 1 can diced tomatoes (14 ounces)
- 1 teaspoon dried thyme
- Salt and black pepper to taste
- 1 cup green beans, chopped

Instructions:

1. Heat the olive oil in a large soup pot over medium heat.
2. Add the onion, carrots, and celery. Cook for approximately five minutes until softened.
3. Stir in the garlic and thyme. Cook for another minute.
4. Add the potatoes, zucchini, green beans, diced tomatoes, and vegetable broth.
5. Bring the mixture to a boil, then reduce the heat.
6. Cover the pot and simmer for 20–25 minutes, until the vegetables are tender.
7. Season with salt and black pepper according to taste.

Serving Suggestions:

Ladle the soup into bowls and garnish with fresh parsley. Serve with warm crusty bread or crackers.

Variations:

For additional protein, add cooked beans or lentils. Small pasta shapes can also be included during the final ten minutes of cooking. Fresh basil or rosemary can replace thyme for a different flavor.
"""