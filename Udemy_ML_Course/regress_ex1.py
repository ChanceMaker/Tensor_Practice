#simple regression example
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def regress():
	x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
	

	y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

	#plt.plot(x_data,y_label,'*')

	#plt.show()

	#y = mx + b
	num1 = np.random.rand()
	num2 = np.random.rand()
	
	m = tf.Variable(num1)
	b = tf.Variable(num2)

	#figuring out error 

	error = 0

	for x,y in zip(x_data,y_label):
		y_hat = m*x + b
		error += (y-y_hat)**2

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
	train = optimizer.minimize(error)

	init  = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		training_steps = 10000

		for i in range(training_steps):
			sess.run(train)
		final_slope,final_intercept = sess.run([m,b])

	#evaluate the result
	x_test = np.linspace(-1,11,10)
	y_pred_plot = final_slope *x_test + final_intercept

	plt.plot(x_test,y_pred_plot,'r')
	plt.plot(x_data,y_label,'*')

	plt.show()


def main():
	regress()
	



if __name__ == "__main__":
	main()

