0. sample data is in "dataset"
   there are images and motor angle data files

1. training cae and save the learning models
 go to "code_ae" and run
 $ python do_cae.py train
 the trained learning models are saved in "results_ae"
 
2. test cae and extract image feature
 go to "code_ae" and run
 $ python do_cae.py test
 the extracted image features and motor angle data are combined and saved in "results_ae" as txt file.
 
3. convert the txt to pickle
 do to "prepare_dataset" and run
 $ python make_dataset.py
 
4. training mtrnn and save the learning models
 go to "code_rnn" and run
 $ python do_rnn.py train
 the learning models are saved in "results_rnn"
 
5. test mtrnn and generate motions and check the flow of latent space value
 go to "code_rnn" and run
 $ python do_rnn.py test
 the png of motor angle trajectories and the flow of latent space value are saved in "results_rnn"
