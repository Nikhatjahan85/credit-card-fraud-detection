import os
import sys
import time
import logging

# ==============================
# LOGGING SETUP
# ==============================
import os
import logging

# ✅ Auto create folder
os.makedirs("outputs", exist_ok=True)

logging.basicConfig(
    filename="outputs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# ==============================
# COLOR FUNCTIONS
# ==============================
def green(text): return f"\033[92m{text}\033[0m"
def red(text): return f"\033[91m{text}\033[0m"
def yellow(text): return f"\033[93m{text}\033[0m"
def blue(text): return f"\033[94m{text}\033[0m"

# ==============================
# CHECK MODEL
# ==============================
def model_exists():
    return os.path.exists("models/model.pkl")

# ==============================
# TRAIN MODEL
# ==============================
def train_model():
    print(blue("\n🚀 Training Model...\n"))
    logging.info("Training started")

    try:
        os.system("python src/train.py")
        logging.info("Training completed")
        print(green("✅ Model trained successfully"))
    except Exception as e:
        logging.error(f"Training error: {e}")
        print(red("❌ Training failed"))

# ==============================
# RUN DASHBOARD
# ==============================
def run_dashboard():
    if not model_exists():
        print(red("\n❌ Model not found! Train model first.\n"))
        return

    print(blue("\n📊 Launching Dashboard...\n"))
    logging.info("Dashboard launched")

    os.system("streamlit run dashboard/app.py")

# ==============================
# VIEW LOGS
# ==============================
def view_logs():
    print(yellow("\n📜 Showing Logs:\n"))

    try:
        with open("outputs/app.log", "r") as f:
            print(f.read())
    except FileNotFoundError:
        print(red("No logs found."))

# ==============================
# MENU
# ==============================
def show_menu():
    print(blue("\n====== 💳 FRAUD DETECTION SYSTEM ======"))
    print("1. Train Model")
    print("2. Run Dashboard")
    print("3. View Logs")
    print("4. Exit")

# ==============================
# MAIN LOOP
# ==============================
if __name__ == "__main__":
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            train_model()
        elif choice == "2":
            run_dashboard()
        elif choice == "3":
            view_logs()
        elif choice == "4":
            print(green("\n👋 Exiting... Goodbye!\n"))
            logging.info("Program exited")
            sys.exit()
        else:
            print(red("\n❌ Invalid choice. Try again.\n"))

        time.sleep(1)