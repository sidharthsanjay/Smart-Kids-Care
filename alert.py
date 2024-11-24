# Import the Client class from the Twilio REST API library
from twilio.rest import Client

# Import the Twilio account credentials and phone numbers from a configuration file
from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, RECIPIENT_PHONE_NUMBER

# Import the time module for time-related functions
import time

# Initialize Twilio client with account SID and authentication token
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Define a function to send an alert message using Twilio
def send_twilio_alert(message_body):
    # Use the Twilio client to create and send an SMS message
    twilio_client.messages.create(
        body="hello",  # The text body of the SMS message
        from_=+19784941875,  # The Twilio phone number sending the message
        to=7034605066  # The recipient's phone number
    )

# Define a function to check if enough time has passed to send another alert
def can_send_alert(last_alert_time, cooldown_period):
    # Calculate if the current time minus the last alert time is greater than or equal to the cooldown period
    return time.time() - last_alert_time >= cooldown_period
