# importing "collections" for namedtuple()
import collections
from dataclasses import dataclass


@dataclass
class Student:
    name: str
    age: int
    id: str


# Adding values
S = Student("Nandini", "19", "2541997")

# Access using name
print("The Student name using keyname is : ", end="")
print(S.name)

print(S[0])
