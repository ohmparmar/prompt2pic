from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.contrib import messages
from django.core.exceptions import ValidationError
import re
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth import (
    authenticate,
    login as auth_login,
    logout,
    get_user_model,
)
from django.contrib.messages import get_messages
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags

CustomUser = get_user_model()
import random

# Create your views here.
from django.template.loader import get_template


def signup(request):
    if request.user.is_authenticated:
        return redirect("image_generation:dashboard")
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")

        # Check if email is already registered
        try:
            user = CustomUser.objects.get(email=email)  # Use CustomUser
            if user.otp_verified:
                messages.error(request, "Email is already registered.")
                return render(
                    request,
                    "authentication/signup.html",
                    {"username": username, "email": email},
                )
            else:
                resend_otp(user)
                messages.info(request, "OTP sent to your email. Please verify.")
                request.session["email"] = email  # Store email in session
                request.session["password"] = password  # Store password in session
                return redirect("authentication:otp_verification")
        except CustomUser.DoesNotExist:
            pass

        # Password validation
        if (
            len(password) < 6
            or not re.search(r"[A-Z]", password)
            or not re.search(r"[0-9]", password)
            or not re.search(r'[!@#$%^&*(),.?":{}|<>]', password)
        ):
            messages.error(
                request,
                "Password must be at least 6 characters long and include one capital letter, one number, and one special character.",
            )
            return render(
                request,
                "authentication/signup.html",
                {"username": username, "email": email},
            )

        hashed_password = make_password(password)  # Hash the password
        otp = str(random.randint(100000, 999999))  # Generate a 6-digit OTP

        # Save user with OTP
        user = CustomUser.objects.create(
            username=username, email=email, password=hashed_password, otp=otp
        )  # Use CustomUser

        # Render email template with OTP
        email_html_content = render_to_string(
            "authentication/email_otp_template.html", {"otp": otp}
        )
        email_text_content = strip_tags(email_html_content)

        # Send HTML email
        email_message = EmailMultiAlternatives(
            subject="Your OTP Code",
            body=email_text_content,  # Plain text version
            from_email="from@example.com",
            to=[email],
        )
        email_message.attach_alternative(email_html_content, "text/html")
        email_message.send()
        messages.success(request, "OTP sent to your email. Please verify.")
        request.session["email"] = email  # Store email in session
        request.session["password"] = password  # Store password in session
        return redirect("authentication:otp_verification")
    return render(request, "authentication/signup.html")


def resend_otp(user):
    otp = str(random.randint(100000, 999999))  # Generate a new OTP
    user.otp = otp
    user.save()
    email_html_content = render_to_string(
        "authentication/email_otp_template.html", {"otp": otp}
    )
    email_text_content = strip_tags(email_html_content)

    # Send HTML email
    email_message = EmailMultiAlternatives(
        subject="Your OTP Code",
        body=email_text_content,  # Plain text version
        from_email="from@example.com",
        to=[user.email],
    )
    email_message.attach_alternative(email_html_content, "text/html")
    email_message.send()

    # send_mail(
    #     'Your OTP Code',
    #     f'Your OTP code is {otp}',
    #     'from@example.com',
    #     [user.email],
    #     fail_silently=False,
    # )


def otp_verification(request):
    email = request.session.get("email")  # Retrieve email from session
    if request.method == "POST":
        if not email:
            messages.error(request, "Email is required to resend OTP.")
            return render(
                request, "authentication/otp_verification.html", {"email": email}
            )

        try:
            user = CustomUser.objects.get(email=email)  # Use CustomUser
        except CustomUser.DoesNotExist:
            messages.error(request, "User not found. Please register first.")
            return render(
                request, "authentication/otp_verification.html", {"email": email}
            )

        if "resend_otp" in request.POST:
            resend_otp(user)
            messages.info(request, "OTP resent successfully.")
            return render(
                request, "authentication/otp_verification.html", {"email": email}
            )

        otp_input = "".join([request.POST.get(f"otp{i}") for i in range(1, 7)])
        if "forgot_password" in request.session:
            del request.session["forgot_password"]
            if user.otp == otp_input:
                return redirect("authentication:reset_password")
            else:
                messages.error(request, "Invalid OTP. Please try again.")
                return render(
                    request, "authentication/otp_verification.html", {"email": email}
                )

        if user.otp == otp_input:
            user.otp_verified = True
            user.otp = None
            user.save()
            # Authenticate the user using the username and password
            password = request.session.get("password")  # Retrieve password from session
            user = authenticate(
                username=user.username, password=password
            )  # Use CustomUser
            if user is not None:
                auth_login(request, user)
                del request.session["email"]
                del request.session["password"]  # Clear password from session
                messages.success(request, "OTP verified successfully!")
                return redirect("image_generation:dashboard")
            else:
                messages.error(request, "Invalid credentials. Please try again.")
        else:
            messages.error(request, "Invalid OTP. Please try again.")
    return render(request, "authentication/otp_verification.html", {"email": email})


def login(request):
    if request.user.is_authenticated:
        return redirect("image_generation:dashboard")
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        # Authenticate the user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            auth_login(request, user)  # Log the user in
            messages.success(request, "Login successful!")
            return redirect("image_generation:dashboard")  # Redirect to the dashboard
        else:
            messages.error(request, "Invalid credentials. Please try again.")
            return render(request, "authentication/login.html")  # Return to login form

    return render(
        request, "authentication/login.html"
    )  # Render the login form for GET requests


def user_logout(request):
    # Log the user out and clear authentication data
    logout(request)

    # Explicitly flush the entire session
    request.session.flush()

    # Clear any previous messages
    storage = get_messages(request)
    for _ in storage:
        pass  # This clears stored messages

    # Add success message for logout
    messages.success(request, "You have been logged out successfully.")
    return redirect("authentication:login")


def forgot_password(request):
    if request.method == "POST":
        email = request.POST.get("email")

        # Validate email format
        if not validate_email(email):
            messages.error(request, "Invalid email format.")
            return render(request, "authentication/forgot_password.html")

        try:
            user = CustomUser.objects.get(email=email)
            if user.otp_verified:  # Check if the user is verified
                otp = str(random.randint(100000, 999999))  # Generate OTP
                user.otp = otp
                user.save()

                # Send OTP email
                email_html_content = render_to_string(
                    "authentication/email_otp_template.html", {"otp": otp}
                )
                email_text_content = strip_tags(email_html_content)

                email_message = EmailMultiAlternatives(
                    subject="Your OTP Code",
                    body=email_text_content,
                    from_email="noreply@prompt2pic.com",
                    to=[email],
                )
                email_message.attach_alternative(email_html_content, "text/html")
                email_message.send()
                request.session["email"] = email
                request.session["forgot_password"] = 1

                messages.success(request, "OTP sent to your email. Please verify.")
                return redirect(
                    "authentication:otp_verification"
                )  # Redirect to OTP verification page
            else:
                messages.error(request, "User is not verified.")
        except CustomUser.DoesNotExist:
            messages.error(request, "User does not exist.")

    return render(request, "authentication/forgot_password.html")


def validate_email(email):
    # Basic email validation logic
    import re

    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(email_regex, email) is not None


def reset_password(request):
    if request.method == "POST":
        new_password = request.POST.get("new_password")
        confirm_password = request.POST.get("confirm_password")

        # Validate password
        if new_password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return render(request, "authentication/reset_password.html")

        if (
            len(new_password) < 6
            or not re.search(r"[A-Z]", new_password)
            or not re.search(r"[a-z]", new_password)
        ):
            messages.error(
                request,
                "Password must be at least 6 characters long and include one capital letter and one small letter.",
            )
            return render(request, "authentication/reset_password.html")

        # Update password logic here

        print(request.session)
        hashed_password = make_password(new_password)  # Hash the password
        user = CustomUser.objects.get(email=request.session["email"])
        user.password = hashed_password
        user.save()

        # user.set_password(new_password)
        # user.save()
        messages.success(request, "Password reset successfully.")
        return redirect(
            "authentication:login"
        )  # Redirect to login after successful reset

    return render(request, "authentication/reset_password.html")
