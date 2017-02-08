# Welcome

# In python, "comments" begin with a "#" sign, so we can use them to leave
# helpful notes for anyone reading our code


# import statements are used to bring in someone else's code library
# here we pull in a "time" library
import time

# The "time" library can be used to get information about the current date
# The syntax can be confusing, but here we use it to set a variable called
#  "today" to the current day of the week
today = time.strftime("%A")


# Here we print out the variable we just set
print("Today is",today)


# First let's set up some variables
monday_activity = "sky diving"
tuesday_activity = "yoga"
wednesday_activity = "golf"
thursday_activity = "running"




# if - elif - else  example
# in python, "elif" means "else if"

if today == "Monday":
  
  print("This evening you're doing",monday_activity)
  
elif  today == "Tuesday":

  print("This evening you're doing",tuesday_activity)

elif  today == "Wednesday":

  print("This evening you're doing",wednesday_activity)
  print("Are you sure that's a real sport?")
  
elif today == "Thursday":

  print("This evening you're doing",thursday_activity)
  print("It is nearly Friday!")
  
else:

  print("You have a free evening!")


