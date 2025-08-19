"""
core/function_calling.py
========================
This module contains backend automation function implementations
for NextGenLingo — the advanced conversational AI system.

These functions are triggered based on detected user intent,
allowing the assistant to perform actions like sending emails,
adding tasks, retrieving data, or controlling external services.
"""

import datetime
import pytz


def execute_action(action_name, params):
    """
    Executes backend actions based on intent and parameters.
    
    :param action_name: The action/intent name (string).
    :param params: Dictionary of parameters for the action.
    :return: String result message.
    """

    # ---- CURRENT DATE ----
    if action_name == "get_date":
        today = datetime.datetime.now().strftime("%A, %B %d, %Y")
        print(f"[Action] Retrieved system date: {today}")
        return f"📅 Today is {today}."

    # ---- CURRENT TIME ----
    elif action_name == "get_time":
        # Returns current time (could be timezone-aware)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Action] Retrieved system time: {now}")
        return f"⏰ Current time: {now}"

    # ---- EMAIL SENDING ----
    elif action_name == "send_email":
        to = params.get("to", "unknown recipient")
        subject = params.get("subject", "No Subject")
        body = params.get("body", "")
        # Simulated action (replace with real email API integration)
        print(f"[Action] Sending email to {to} | Subject: '{subject}' | Body: {body}")
        return f"✅ Email sent to {to} with subject '{subject}'."

    # ---- ADD TO-DO TASK ----
    elif action_name == "add_todo":
        task = params.get("task", "Unnamed task")
        # Simulated action (replace with real to-do system integration)
        print(f"[Action] Adding task to to-do list: {task}")
        return f"📝 Task '{task}' added to your to-do list."

    # ---- WEATHER INFORMATION ----
    elif action_name == "weather":
        location = params.get("location", "your area")
        # Simulated response (replace with actual weather API call)
        print(f"[Action] Fetching weather for {location}")
        return f"☀ The weather in {location} is sunny and warm today."

    # ---- FALLBACK ----
    else:
        print(f"[Error] Unknown action: {action_name}")
        return "❌ Sorry, I don't recognize that action."


# Standalone test cases
if __name__ == "__main__":
    print(execute_action("send_email", {
        "to": "user@example.com",
        "subject": "Meeting Reminder",
        "body": "Don't forget our 3 PM meeting today."
    }))
    
    print(execute_action("add_todo", {
        "task": "Finish the project report"
    }))
    
    print(execute_action("weather", {
        "location": "New York"
    }))

    print(execute_action("get_time", {}))

    print(execute_action("get_date", {}))

    print(execute_action("unknown_action", {}))
