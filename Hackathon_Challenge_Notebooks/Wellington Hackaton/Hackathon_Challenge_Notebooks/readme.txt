############################################################################
# Wellington Campus x DAFZ AI Hackathon 2024 - Participant Setup Guide
############################################################################

Welcome, Hackathon Participants!

This guide will walk you through setting up your computer to generate the
mock datasets needed for your specific hackathon challenge. We'll use Python,
virtual environments, and Jupyter Notebooks.

Follow these steps carefully. If you get stuck, please ask a mentor or
visit the Help Desk!

**Estimated Time:** 15-25 minutes

------------------------------------
### Step 0: Prerequisites - Check Python Installation
------------------------------------

You need Python installed on your system. Version 3.8 or higher is recommended.
To check if you have Python and its version, open your Terminal (macOS/Linux)
or Command Prompt/PowerShell (Windows) and type ONE of the following commands:

```bash
python --version
```

OR

```bash
python3 --version
```

If you see a version number (like `Python 3.9.7`), you're good! If not, or if
the version is older than 3.8, please download and install the latest stable
Python 3 version from the official website:

*   Python Downloads: https://www.python.org/downloads/

Make sure to check the box that says "Add Python to PATH" during installation
on Windows if available.

------------------------------------
### Step 1: Download and Unzip Hackathon Files
------------------------------------

1.  You should have received a ZIP file containing the hackathon materials,
    likely named something like `Hackathon_Challenge_Notebooks.zip`.
2.  Download this ZIP file to a location you can easily access (e.g., your
    Desktop or a dedicated `Projects` folder).
3.  Unzip the file. You should now have a folder named
    `Hackathon_Challenge_Notebooks`. This is your main project folder.

------------------------------------
### Step 2: Create Project Sub-Directories (Using Provided Script)
------------------------------------

We've included a Python script to automatically create the necessary
sub-folders for each challenge.

1.  **Open your Terminal or Command Prompt.**
2.  **Navigate** into the main `Hackathon_Challenge_Notebooks` folder you just
    unzipped. Use the `cd` (change directory) command:

    ```bash
    # Replace 'path/to/your/unzipped/folder' with the actual path
    cd path/to/your/unzipped/folder/Hackathon_Challenge_Notebooks
    ```
    *Example Windows:* `cd C:\Users\YourName\Desktop\Hackathon_Challenge_Notebooks`
    *Example macOS/Linux:* `cd ~/Desktop/Hackathon_Challenge_Notebooks`

3.  **Run the Python script** to create the folders:

    ```bash
    python create_hackathon_dirs.py
    ```
    *   If `python` doesn't work, try `python3 create_hackathon_dirs.py`.

You should see messages indicating that directories like `Challenge1_Inventory`,
`Challenge2_Routing`, etc., and their `data` subfolders are being created.

------------------------------------
### Step 3: Set Up a Virtual Environment
------------------------------------

A virtual environment keeps the Python packages for this hackathon separate
from other projects on your computer. This prevents conflicts.

1.  **Make sure you are still inside the `Hackathon_Challenge_Notebooks`**
    directory in your terminal (from Step 2).
2.  **Create the virtual environment.** We'll name it `venv`:

    ```bash
    python -m venv venv
    ```
    *   Again, use `python3` if `python` doesn't work.
    *   This command creates a new folder named `venv` inside your project directory.

3.  **Activate the virtual environment.** The command depends on your operating
    system and terminal:

    *   **Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate
        ```
    *   **Windows (PowerShell or Git Bash):**
        ```bash
        source venv/Scripts/activate
        ```
    *   **macOS / Linux (Bash/Zsh):**
        ```bash
        source venv/bin/activate
        ```

4.  **Verify activation:** Your terminal prompt should now start with `(venv)`,
    like this: `(venv) C:\path\to\Hackathon_Challenge_Notebooks>`
    This means the virtual environment is active!

    *Important:* Keep this terminal window open and the environment active for the next steps.

------------------------------------
### Step 4: Install Required Python Packages
------------------------------------

With the virtual environment (`venv`) active, we can now install the necessary
Python libraries.

1.  In the same terminal where `(venv)` is active, run the following command:

    ```bash
    pip install pandas numpy faker jupyterlab
    ```
    *   `pip` is Python's package installer.
    *   This command downloads and installs:
        *   `pandas`: For working with data tables (DataFrames).
        *   `numpy`: For numerical calculations.
        *   `faker`: To generate realistic mock data.
        *   `jupyterlab`: The interface we'll use to run the notebooks.

2.  Wait for the installation to complete. You might see a lot of text output.
    Look for a success message at the end.

------------------------------------
### Step 5: Launch JupyterLab
------------------------------------

JupyterLab provides a web-based interface for running code notebooks.

1.  Make sure your virtual environment `(venv)` is still active in the terminal.
2.  Make sure you are still in the main `Hackathon_Challenge_Notebooks` directory.
3.  Run this command to start the JupyterLab server:

    ```bash
    jupyter lab
    ```

4.  This command might take a few seconds. It should automatically open a new tab
    in your default web browser displaying the JupyterLab interface.
5.  If it doesn't open automatically, look at the output in your terminal. You'll
    see lines like:
    `To access the server, open this file in your browser:`
    `file:///C:/Users/.../jupyter_server_...`
    `Or copy and paste one of these URLs:`
    `http://localhost:8888/lab?token=...`  <- Copy this URL
    `http://127.0.0.1:8888/lab?token=...`
    Copy one of the `http://localhost...` or `http://127.0.0.1...` URLs and
    paste it into your web browser's address bar.

6.  **Keep the terminal window running!** Closing it will shut down the
    JupyterLab server.

------------------------------------
### Step 6: Generate Your Challenge Dataset
------------------------------------

Now you'll use the JupyterLab interface to run the specific notebook for
YOUR assigned hackathon challenge.

1.  **Navigate in JupyterLab:** In the left-hand file browser panel within
    JupyterLab (in your web browser), you'll see the folders we created
    (`Challenge1_Inventory`, `Challenge2_Routing`, etc.).
2.  **Find Your Folder:** Double-click on the folder corresponding to the challenge
    you were assigned (e.g., if you got Challenge 1, double-click on
    `Challenge1_Inventory`).
3.  **Open the Notebook:** Inside that folder, you'll see a file ending in
    `.ipynb` (e.g., `Challenge1_Inventory_DataGen.ipynb`). Double-click this
    file to open it in the main work area.
4.  **Read the Setup (Optional):** The notebook itself contains a copy of the
    setup guide you've just followed. You can skip reading that section again.
5.  **Run the Code Cells:**
    *   A notebook is made of "cells". Code cells have `In [ ]:` next to them.
    *   Click on the first code cell (usually the one importing libraries).
    *   Press `Shift + Enter` on your keyboard to run the current cell and
        automatically move to the next one.
    *   Alternatively, you can click the "Run" button (often a triangle icon ▶)
        in the toolbar at the top of the notebook.
    *   **Important:** Run all the code cells in order, from the top of the
        notebook to the bottom, by repeatedly pressing `Shift + Enter`.
    *   You will see output below some cells, including messages like
        "Generating warehouses...", "Saving datasets...", etc. Wait for each
        cell to finish (the `In [*]:` will change to `In [number]:`) before
        running the next.
6.  **Wait for Completion:** The final cells will save the data files and might
    display the first few rows of the generated datasets for verification. Look
    for a message like "data generation complete".

------------------------------------
### Step 7: Locate Your Generated Data Files
------------------------------------

Success! The mock data CSV files specific to your challenge have been generated.

1.  In the JupyterLab file browser (left panel), look inside the **`data`**
    subfolder within *your specific challenge folder* (e.g., inside
    `Hackathon_Challenge_Notebooks/Challenge1_Inventory/data/`).
2.  You will find the `.csv` files listed there (e.g., `warehouses.csv`,
    `inventory.csv`, etc.).
3.  You will also find a `README.md` file in that `data` folder explaining the
    columns in your specific CSV files.

------------------------------------
### Step 8: Start Hacking!
------------------------------------

You now have:
*   A working Python environment (`venv`).
*   JupyterLab installed and running.
*   The specific mock datasets (`.csv` files) for your challenge.

You can now:
*   Create a *new* Jupyter Notebook within your challenge folder (File -> New -> Notebook) to start coding your solution.
*   Load the CSV data into your new notebook using `pandas` (e.g., `pd.read_csv('data/inventory.csv')`).
*   Begin developing your AI model or logic!

Remember to consult the main Hackathon Guide document for the problem statement details, deliverables, and evaluation criteria.

------------------------------------
### Step 9: Shutting Down (When Finished for the Day)
------------------------------------

1.  **Save your work** in any Jupyter notebooks you created.
2.  **Shutdown JupyterLab:** Go back to the **Terminal window** where you ran
    `jupyter lab`. Press `Ctrl + C` (hold Ctrl and press C). It might ask you
    to confirm shutdown; press `y` and Enter if needed.
3.  **Deactivate the virtual environment:** In the same terminal, type:
    ```bash
    deactivate
    ```
    The `(venv)` prefix should disappear from your prompt.
4.  You can now close the terminal window.

To start working again later, just navigate back to the
`Hackathon_Challenge_Notebooks` folder in your terminal, activate the
environment (`source venv/bin/activate` or `venv\Scripts\activate`), and
run `jupyter lab`.

------------------------------------
### Troubleshooting Tips
------------------------------------

*   **`python` or `pip` command not found:** Ensure Python is installed correctly
    and added to your system PATH. Try using `python3` and `pip3` instead.
*   **`ModuleNotFoundError` (e.g., "No module named 'pandas'"):** This usually
    means you forgot to activate the virtual environment (`venv`) before running
    `pip install` or `jupyter lab`. Deactivate (`deactivate`), reactivate
    (`source venv/bin/activate` or `venv\Scripts\activate`), and try installing
    again (`pip install ...`).
*   **JupyterLab doesn't open / Forbidden / Errors in terminal:** Check the
    terminal output carefully for error messages. Ensure no other program is using
    the same port (usually 8888). Try restarting the command. Copy the full URL
    from the terminal into your browser.
*   **Permission Errors during `pip install`:** Avoid using `sudo` with `pip`
    inside a virtual environment. If you encounter permission issues *outside*
    the `venv` folder (unlikely for this setup), consult a mentor.
*   **Other Issues:** Don't hesitate to ask the mentors or help desk staff!

Good luck, have fun, and build something amazing!
############################################################################

