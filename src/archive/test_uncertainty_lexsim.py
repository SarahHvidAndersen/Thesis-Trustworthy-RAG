#from lm_polygraph.estimators import LexicalSimilarity
from uncertainty_estimation.lexical_similarity import LexicalSimilarity
import requests
import os
import numpy as np

#Near -1.0 → low uncertainty (identical/similar)
#between -0.5 – -0.8 → moderate
#Below -0.5 → high uncertainty
# should be calibrated for my use case

CHATUI_API = os.getenv("CHATUI_API_URL")
#MODEL = "llama3:8b"
MODEL = "llama3.2:1b"
N_SAMPLES = 2


def generate_response(prompt: str, temperature=0.3) -> str:

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": None
        }
    }
    response = requests.post(CHATUI_API, json=payload)
    response.raise_for_status()
    return response.json().get("response", "").strip()

def generate_samples(prompt: str, n: int) -> list:
    samples = []
    for i in range(n):
        print(f"Generating sample {i+1}/{n}...")
        reply = generate_response(prompt)
        samples.append(reply)
    return samples

def estimate_uncertainty_from_samples(samples: list) -> np.ndarray:
    estimator = LexicalSimilarity(metric="rougeL")
    result = estimator({"sample_texts": [samples]})
    return result  # returns np.array


def main(test):
    if test:
        #test_case = "long_similar"  # Options: "short_similar", "short_dissimilar", "long_similar"
        test_case = "short_dissimilar"

        if test_case == "short_similar":
            samples = [
                "A cat is a domesticated carnivorous mammal.",
                "A cat is a small domesticated carnivorous mammal.",
                "A cat is a small carnivorous mammal.",
                "A cat is a small domesticated carnivorous mammal.",
                "A cat is a small domesticated carnivorous mammal."
            ]
        elif test_case == "short_dissimilar":
            samples = [
                "A cat is a dog.",
                "A dog is a reptile.",
                "A computer is a planet.",
                "A table is a programming language.",
                "A lamp is an element on the periodic table."
            ]
        elif test_case == "long_similar":
            samples = [
                """A question that gets to the purr-fectly fundamental level!

A cat, scientifically known as Felis catus, is a small, typically furry, carnivorous mammal. Cats are members of the family Felidae and are characterized by their agility, playful personalities, and ability to purr (a unique vocalization). They are often kept as pets or companions due to their affectionate nature and ability to adapt to human environments.

Here are some key characteristics that define a cat:
1. **Physical appearance**: Cats have a slender body with four legs, a tail, ears, and eyes. They typically range in size from 8-12 inches (20-30 cm) in length, including the tail.
2. **Diet**: Cats are carnivores, which means they primarily feed on animal-based foods such as meat, fish, or insects. In captivity, they are often fed commercial cat food.
3. **Behavior**: Cats are known for their independence, playful nature, and curious behavior. They spend a significant amount of time sleeping, grooming themselves, and engaging in activities like hunting, exploring, and socializing with humans.
4. **Communication**: Besides purring, cats use vocalizations (meowing, hissing), body language (ear positions, tail movements), and scent marking to communicate with each other and humans.
5. **Reproduction**: Female cats give birth to a litter of kittens after a gestation period of approximately 63-65 days. The average litter size is two to five kittens.

With over 70 recognized breeds, cats have become an integral part of many cultures and households worldwide.""",
                """A cat, also known as Felis catus, is a small, typically furry, carnivorous mammal that belongs to the family Felidae. Cats are one of the most popular pets in the world and have been domesticated for thousands of years.

Physical Characteristics:
* Cats are usually between 8-12 inches (20-30 cm) long, including their tail.
* They weigh around 8-15 pounds (3.6-6.8 kg), although some breeds can be much larger or smaller.
* Their bodies are typically covered with a thin layer of fur, which can vary in color and texture depending on the breed.
* Cats have a distinctive head shape, with ears that are usually pointed and eyes that are almond-shaped.
* They have retractable claws, which they use for climbing, hunting, and self-defense.

Behavior:
* Cats are known for their independence and aloofness. They often spend most of their time sleeping or grooming themselves.
* Despite their independent nature, cats can be very affectionate and playful, especially with their human companions.
* They are naturally curious and enjoy exploring their surroundings.
* Cats are predators and have a strong hunting instinct, which they use to catch small animals like mice, birds, and insects.

Types of Cats:
* Domestic Shorthair: A generic term for cats that do not belong to specific breeds, but are still considered pets.
* Siamese: Originating from Thailand, this breed is known for its striking blue eyes and pointy ears.
* Persian: One of the most recognizable cat breeds, characterized by their long hair, flat faces, and calm demeanor.
* Maine Coon: A large and fluffy breed with a distinctive shaggy coat and bushy tail.

Overall, cats are unique animals that have captured the hearts and imaginations of many people around the world.""",
                """A question that has puzzled humans for centuries!

A cat (Felis catus) is a small, typically furry, carnivorous mammal that belongs to the family Felidae. They are one of the most popular pets in the world and are known for their agility, playfulness, and affectionate nature.

Here are some key characteristics that define what makes a cat:
1. **Physical appearance**: Cats have a slender body, usually with a furry coat, pointed ears, and whiskers. Their fur can be short or long, depending on the breed.
2. **Diet**: Cats are obligate carnivores, which means they primarily feed on meat (like mice, birds, or commercial cat food). They have specialized teeth and digestive systems designed for consuming animal tissue.
3. **Behavior**: Cats are known for their playful, curious nature. They like to chase toys, climb, and scratch surfaces (which can sometimes be destructive).
4. **Sleep patterns**: Cats spend a significant amount of time sleeping or lounging, often in short intervals throughout the day. They have a unique sleep pattern called "catnapping," where they rest while still remaining alert for potential prey.
5. **Communication**: Cats communicate primarily through body language (like ear and tail positions) and vocalizations (such as meows, purrs, and hisses). They can also use scent marking to convey information.

Some interesting cat facts:
* There are over 70 recognized cat breeds worldwide, each with unique characteristics.
* Cats have retractable claws that they use for climbing, hunting, and self-defense.
* They have a highly developed sense of hearing and vision, allowing them to detect prey or potential threats from a distance.
* Cats are known for their grooming habits, using their tongues to clean their fur and remove parasites.

Overall, cats are fascinating creatures with distinct characteristics that make them beloved companions and important members of many households.""",
                """A simple yet fascinating question!

A cat, also known as Felis catus, is a small, typically furry, carnivorous mammal. Cats are members of the family Felidae and are closely related to big cats, such as lions, tigers, and leopards.

Here are some key characteristics that define what a cat is:
1. **Physical appearance**: Cats have a slender body, usually with a rounded head, pointed ears, whiskers (long hairs around their mouth), and a long tail.
2. **Furry coat**: Most cats have a soft, thick fur that can vary in color and pattern, from solid colors to stripes, dots, or swirling patterns.
3. **Carnivorous diet**: Cats are meat-eaters, primarily feeding on small animals like mice, birds, insects, and other small creatures. They have specialized teeth and jaws for tearing flesh and crushing bones.
4. **Independence**: Cats are known for their independence and aloofness, often spending time alone or sleeping a lot.
5. **Agility and hunting skills**: Cats are agile and stealthy hunters, using their sharp claws, flexible spine, and keen senses (vision, hearing, and smell) to catch prey.
6. **Social behavior**: While individual cats may be independent, they can also form strong bonds with humans or other felines, and engage in social behaviors like purring, grooming, and playful interactions.

There are over 70 recognized breeds of domestic cats, each with their unique characteristics, sizes, and personalities. So, that's what a cat is – a fascinating creature with its own special features and behaviors!""",
                """A cat, also known as Felis catus, is a domesticated mammal and one of the most popular pets in the world. Cats belong to the family Felidae and are closely related to big cats like lions, tigers, and leopards.

Here are some key characteristics that define a cat:
1. **Physical appearance**: Cats have a slender body, typically with a length of around 10-12 inches (25-30 cm) and a weight ranging from 8-15 pounds (3.5-6.8 kg). They have four legs, a flexible spine, and sharp claws. Their fur can vary in color and pattern, from solid colors to striped or spotted patterns.
2. **Diet**: Cats are obligate carnivores, meaning they primarily feed on animal-based foods such as meat, fish, and insects. In the wild, they hunt small prey like mice, birds, and other small animals.
3. **Behavior**: Cats are known for their independent nature, curiosity, and playful behavior. They are skilled predators, using their acute senses (hearing, vision, smell) to stalk and pounce on prey. Domesticated cats often exhibit more docile behavior, enjoying human interaction and affection.
4. **Communication**: Cats communicate primarily through body language (posture, facial expressions, tail positions), vocalizations (meowing, purring, hissing), and scent marking.
5. **Lifespan**: The average lifespan of a domesticated cat is around 12-15 years, depending on factors like diet, health, and lifestyle.

Cats have been human companions for thousands of years, with evidence suggesting that they were first domesticated in ancient Egypt around 4,000 years ago. Today, there are over 600 million domestic cats worldwide, making them one of the most popular pets globally!"""
            ]
        else:
            raise ValueError("Invalid test_case value")

    else:
        prompt = 'Who is the president of denmark? tell me about them in two sentences.'
        samples = generate_samples(prompt, N_SAMPLES)

    print("\n=== Generated Samples ===")
    for i, s in enumerate(samples):
        print(f"[{i+1}] {s}\n")

    score_array = estimate_uncertainty_from_samples(samples)
    print(score_array)
    final_score = float(score_array[0])
    #negated_final_score = -score_array[0]

    print("\n=== Final Uncertainty Score ===")
    print(final_score)
    #print(negated_final_score)


def main(test=True):
    if test:
        test_case = "new_dissimilar"  # Options: "new_similar", "new_dissimilar"

        if test_case == "new_similar":
            # Five reworded definitions of Newton's second law (F = m*a)
            samples = [
                "Newton's second law states that the force acting on an object is equal to the mass of the object multiplied by its acceleration.",
                "According to Newton's second law, F equals the product of mass and acceleration, meaning that an object's acceleration is directly proportional to the net force acting on it.",
                "In physics, Newton's second law defines force as mass times acceleration (F = m * a), which explains how an object's motion changes when a force is applied.",
                "The principle of Newton's second law tells us that force is equal to mass multiplied by acceleration, summarizing the relationship F = m*a.",
                "Newton's second law of motion establishes that the net force acting on a body is the product of its mass and its acceleration."
            ]
        elif test_case == "new_dissimilar":
            samples = [
                "Bananas",
                "The Great Wall of China",
                "Pythagoras.",
                "Mount Everest",
                "Classical music."
            ]
        else:
            raise ValueError("Invalid test_case value")
    else:
        raise NotImplementedError("Non-test mode is not implemented in this snippet.")

    print("\n=== Generated Samples ===")
    for i, s in enumerate(samples):
        print(f"[{i+1}] {s}\n")
    
    score_array = estimate_uncertainty_from_samples(samples)
    print("\nRaw uncertainty scores (lexsim estimator):")
    print(score_array)
    final_score = float(score_array[0])
    print("\n=== Final Uncertainty Score for Test Case ===")
    print(final_score)

if __name__ == "__main__":
    main(test=True)
