from predict_lora import Predictor
obj = Predictor()

obj.predict_pose(image='./testData/test9.jpg', sample_algorithm='ddim', num_poses=3, nsteps=50)