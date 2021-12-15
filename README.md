# BDC_test
This is a test task for vacancy of junior data scientist 

Setup guide
1. Clone this repository to your local machine
2. Create new virtual environment with requirements.txt
3. Download image data from  https://drive.google.com/file/d/1RDAeRaX526vcZ4PcZnYqSFqsibmxMdmd/view and unpack to BDC_test/data/dataset
4. Generate dataset using notebook BDC_test/source/modelling/train_step1.ipynb/ You should get train.csv, test.csv, vslid.csv at BDC_test/data/dataset
5. Configure your ClearML account
6. Use train.py to launch training. My results you can see at https://app.community.clear.ml/projects/*/experiments/7623601c07344f81ae9a6a0dc2307b77/info-output/metrics/scalar?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&deep=true
7. To run an inference, launch BDC_test/source/inference/inference_tryout.ipynb notebook