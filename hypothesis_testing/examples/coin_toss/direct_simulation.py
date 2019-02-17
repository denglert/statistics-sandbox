from numpy.random import randint

# - Direct simulation

# - Problem:
# - You toss a coin 30 times and see 22 heads and 8 tails.
#
# - Question:
# - Is this a fair coin?
#
# - Null hypothesis (H0):
# - Fair coin:
# - P(head) = 0.5  
# - P(tail) = 0.5  
#
# - Test of hypothesis H0
# - p-value = P( heads > 22 )
# - Reject fair coin hypothesis at p-value < 0.05
# 
# - What is the probability of observing this sum,
# - N(head) = 22, N(tail) = 8, or an even larger number
# - of heads?
# - P( heads > 22 )


# - Simulation
#
# - head = 1
# - tail = 0
# 
# - Single experiment:
# - Generate binary number (0 or 1) 30 times
#
# - Repeat this experiment 10000 times to get a distribution


# - Count the number of experiments where the we have 22 or more heads
M = 0

# - nExp experiments
nExp = 100000

for i in range ( nExp ):

    # - Generate list of random binaries of size 30
    trials = randint(2, size=30)

    # - If the sum of the trials (= number of heads) is equal or more than 22
    if ( trials.sum() >= 22):

        # - We increase the count
        M += 1

p_value = M/nExp


###############################

print("Last sequence trial:")
print( trials )

print()

print("Total number of experiments:           {}".format(nExp) )
print("Number of experiments with N(h) >= 22: {}".format(M) )

print()

p_value_lim = 0.05

print("p-value limit:      {:3e}".format(p_value_lim) )
print("Calculated p-value: {:3e}".format(p_value ) )

if p_value < p_value_lim:
    print("p value is smaller than p-value limit. H0 excluded.")
else:
    print("p value is larger than p-value limit. H0 non-excluded.")

print()
