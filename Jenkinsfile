pipeline {
    agent any

    stages {
        stage('Getting Project from Git') {
            steps {
                echo 'Project is downloading...'
                git branch: 'master', url: 'https://github.com/vijay-jayaraman/jenkins-mlflow-docker.git'
            }
        }
        stage('Building Docker Container') {
            steps {
                bat 'docker build -t heartdisease-model .'
                bat 'docker run -d --name model heartdisease-model'
            }
        }
        stage('Preprocessing Stage') {
            steps {
                bat 'docker container exec model python3 preprocessing.py'
            }
        }
        stage('Training Stage') {
            steps {
                bat 'docker container exec model python3 train.py'
            }
        }
        stage('Manual Approval for Model Registration') {
            steps {
                script {
                    input message: "Model training completed. Please register your ML model and approve to proceed to testing."
                }
            }
        }
        stage('Test Stage') {
            steps {
                bat 'docker container exec model python3 test.py'
                bat 'docker rm -f model'
            }
        }
    }
}
