# Input values for age and nationality
age = int(input("Enter your age: "))
nationality = input("Are you a citizen? (yes/no): ")

# Check conditions using logical operators
if age > 18 and nationality == "yes":
    print("You are eligible for voting.")
elif age < 18:
    print("You are a minor.")
elif nationality == "no":
    print("You are not a citizen.")
else:
    print("your are not able for voting,right now.")



