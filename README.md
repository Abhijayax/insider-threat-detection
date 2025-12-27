🔐 Insider Threat Detection
Unsupervised machine learning system to detect malicious insiders by analyzing behavioral patterns in user activity logs.
🎯 Problem
Insider threats cause 60% of data breaches. Traditional security tools can't detect malicious employees who have legitimate access. This system uses ML to automatically flag suspicious behavior.
💡 Solution

Tracks 15 behavioral features (logins, file access, emails, USB usage)
Uses Isolation Forest to detect anomalies
Ranks users by risk score for investigation

📊 Results

80% Precision@10 - 8 out of 10 flagged users are actual threats
20% False Positive Rate - Acceptable for security applications
90% Workload Reduction - Focus on top 10 instead of 1000 users

🚀 Quick Start
bash# Install dependencies
pip install -r requirements.txt

# Train model
python src/train_models.py --model isolation_forest --data data/features/

# Evaluate
python src/evaluate.py --model models/isolation_forest.pkl --data data/features/

# Run dashboard (optional)
streamlit run visualization/dashboard.py
📁 Project Structure
insider-threat-detection/
├── src/                    # Python scripts
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── evaluate.py
├── visualization/          # Dashboard
├── models/                 # Trained models
├── data/                   # Data files
└── requirements.txt
🔧 Features Tracked
Authentication: Login count, off-hours ratio, failed logins, unique machines
File Access: Files accessed, rare files, access spikes, sensitive files
Communication: Email volume, new recipients, after-hours emails
Devices: USB insertions, data copied, new devices
📈 Models

Isolation Forest ⭐ (Best: 80% precision)
One-Class SVM (70% precision)
Autoencoder (65% precision)

📝 Dataset
CERT Insider Threat Dataset - Contains login logs, file access, emails, and USB activity from 1000+ users.
📄 License
MIT License - See LICENSE file
👤 Author
[Abhijaya Singh] - [abhijayax@gmail.com]

⭐ Star this repo if you find it helpful!

