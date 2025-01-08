import random

def guess_the_number():
    # Generate a random number between 1 and 100
    secret_number = random.randint(1, 100)
    attempts = 0
    guessed = False

    print("Welcome to the Guess the Number Game!")
    print("I'm thinking of a number between 1 and 100.")

    while not guessed:
        try:
            # Get user input
            guess = int(input("Enter your guess: "))
            attempts += 1

            # Compare the guess to the secret number
            if guess < secret_number:
                print("Too low! Try again.")
            elif guess > secret_number:
                print("Too high! Try again.")
            else:
                print(f"Congratulations! You guessed the number in {attempts} attempts.")
                guessed = True
        except ValueError:
            # Handle the case where the user doesn't enter a valid integer
            print("Invalid input. Please enter a valid integer.")

if __name__ == "__main__":
    guess_the_number()
