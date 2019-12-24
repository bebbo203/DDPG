from actor import Actor
from critic import Critic



a = Actor(2, 1, 0, 1, 0.005, 0.01)
c = Critic(1,1,0,1,0.005, 0.01 )

s1 = [
    [0., 0.],
    [1., 1.],
    [0., 0.],
    [1., 1.],
]

s3 = [
    [0.],
    [1.],
    [0.],
    [1.],
]

s2 = [
    [0.],
    [0.],
    [1.],
    [1.],
]

truth = [
    [0.],
    [-1.],
    [-1.],
    [0.],
]


for i in range(1000):
    print(c.train(s2, s3, truth).numpy())


print(c.predict(s2, s3))

