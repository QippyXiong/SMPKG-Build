class Logger:
	def __init__(self):
		self.num = 0

	def info(*args):
		print(*args)

	def error(*args):
		print(*args)

	def log_train(self, epoch: int, step: int, loss: float):
		if epoch == 20000:
			self.num += 1
			print(f"Config{self.num} final loss: {loss}")
	
logger = Logger()