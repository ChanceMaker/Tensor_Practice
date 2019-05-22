import numpy as np
import tensorflow as tf

def main():
	np.random.seed(101)
	tf.set_random_seed(101)

	#creating random data points
	rand_a = np.random.uniform(0,100,(5,5))

	#printing out data
	print(rand_a)

	#creating random data with a different shape

	rand_b = np.random.uniform(0,100,(5,1))

	print(rand_b)

	a = tf.placeholder(tf.float32)
	b = tf.placeholder(tf.float32)

	add_op = a + b

	mul_op = a * b

	with tf.Session() as sess:
		add_result = sess.run(add_op,feed_dict={a:rand_a,b:rand_b})
		print(add_result)
		print("---------")
		mult_result = sess.run(mul_op,feed_dict={a:rand_a,b:rand_b})
		print(mult_result)


if __name__ == '__main__':
	main()