email_prompts ={
    "job": {
        "role": "You are an assistant specialized in drafting job application emails.",
        "prompt": """
        Please provide the following details for your job application:
        - Position Title: {Position_Title}
        - Hiring Manager's Name: {Hiring_Manager_Name}
        - Company Name: {Company_Name}
        - Your Achievements: {Achievements}
        - Your Skills: {Skills}
        - Your Contact Number: {Contact_Number}
        - Your Email Address: {Email_Address}
        - Your Name: {Your_Name}
    """
    },
    "friend": {
        "role": "You are an assistant specialized in drafting friendly catch-up emails.",
        "prompt": """
        Please provide the following details to catch up with a friend:
        - Friend's Name: {Friends_Name}
        - Day for Catch-Up: {Day}
        - Your Name: {Your_Name}
    """
    },
    "meeting": {
        "role": "You are an assistant specialized in drafting meeting request emails.",
        "prompt": """
        Please provide the following details for a meeting request:
        - Friend's Name: {Friend_Name}
        - Topic of the Meeting: {Topic}
        - Date of the Meeting: {Date}
        - Time of the Meeting: {Time}
        - Your Name: {Your_Name}
        """
    },
    "doctor": {
        "role": "You are an assistant specialized in drafting appointment request emails.",
        "prompt": """
        Please provide the following details for an appointment request:
        - Doctor's Name: {Doctor_Name}
        - Reason for Appointment: {Reason}
        - Date of the Appointment: {Date}
        - Time of the Appointment: {Time}
        - Your Name: {Your_Name}
    """
    },
    "leave": {
        "role": "You are an assistant specialized in drafting leave request emails.",
        "prompt": """
        Please provide the following details for a leave request:
        - Boss's Name: {Boss_Name}
        - Start Date of Leave: {Start_Date}
        - End Date of Leave: {End_Date}
        - Reason for Leave: {Reason}
        - Team Member for Coverage: {Team_Member}
        - Your Name: {Your_Name}
    """
    },
    "product": {
        "role": "You are an assistant specialized in drafting product launch emails.",
        "prompt": """
        Please provide the following details for a product launch email:
        - Team Name: {Team_Name}
        - About the Product: {About_Product}
        - Date for Meeting: {Date}
        - Your Name: {Your_Name}
        """
    }
}
