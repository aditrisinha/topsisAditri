UCS633
Project 1


Submitted By: Aditri Sinha
Roll Number :101703034


# What is TOPSIS

Technique for Order Preference by Similarity to Ideal

TOPSIS chooses the alternative of shortest Euclidean distance from the ideal solution, and greatest distance from the negative-ideal
solution.

Resources:
https://en.wikipedia.org/wiki/TOPSIS
https://www.youtube.com/watch?v=aRBdrCB1K4k



### In Command Prompt
```
>> topsis data.csv "1,1,1,1" "+,+,-,+"
```

## Example

The decision matrix (`a`) should be constructed with each row representing a Model alternative, and each column representing a criterion like Accuracy, R<sup>2</sup>, Root Mean Squared Error, Correlation, and many more.

Attribute 	price	storage	camera	looks
m1	         250	    16	     12	     5
m2	         200	    16  	  8	     4
m3	         300	    32	     16	     3
m4	         275	    32	      8	     3
m5	         225	    16	     16	     1


<br>
Using TOPSIS the rankings are displayed in the form of a table using a package 'tabulate', with the 1st rank offering the best decision, and last rank offering the worst decision making.
