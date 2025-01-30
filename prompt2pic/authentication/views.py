from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.contrib import messages
from django.core.exceptions import ValidationError
import re
from django.contrib.auth.hashers import make_password,check_password
from django.contrib.auth import authenticate, login as auth_login, logout, get_user_model
CustomUser = get_user_model()
import random

# Create your views here.

def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Check if email is already registered
        try:
            user = CustomUser.objects.get(email=email)  # Use CustomUser
            if user.otp_verified:
                messages.error(request, 'Email is already registered.')
                return render(request, 'authentication/signup.html', {'username': username, 'email': email})
            else:
                resend_otp(user)
                messages.info(request, 'OTP sent to your email. Please verify.')
                request.session['email'] = email  # Store email in session
                request.session['password'] = password  # Store password in session
                return redirect('authentication:otp_verification')
        except CustomUser.DoesNotExist:
            pass

        # Password validation
        if len(password) < 6 or not re.search(r'[A-Z]', password) or not re.search(r'[0-9]', password) or not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            messages.error(request, 'Password must be at least 6 characters long and include one capital letter, one number, and one special character.')
            return render(request, 'authentication/signup.html', {'username': username, 'email': email})

        hashed_password = make_password(password)  # Hash the password
        otp = str(random.randint(100000, 999999))  # Generate a 6-digit OTP
        
        # Save user with OTP
        user = CustomUser.objects.create(username=username, email=email, password=hashed_password, otp=otp)  # Use CustomUser
        
        # Send OTP to user's email
        send_mail(
            'Your OTP Code',
            f'Your OTP code is {otp}',
            'from@example.com',
            [email],
            fail_silently=False,
        )
        messages.success(request, 'OTP sent to your email. Please verify.')
        request.session['email'] = email  # Store email in session
        request.session['password'] = password  # Store password in session
        return redirect('authentication:otp_verification')
    return render(request, 'authentication/signup.html')


def resend_otp(user):
    otp = str(random.randint(100000, 999999))  # Generate a new OTP
    user.otp = otp
    user.save()
    send_mail(
        'Your OTP Code',
        f'Your OTP code is {otp}',
        'from@example.com',
        [user.email],
        fail_silently=False,
    )


def otp_verification(request):
    email = request.session.get('email')  # Retrieve email from session
    if request.method == 'POST':
        if not email:
            messages.error(request, 'Email is required to resend OTP.')
            return render(request, 'authentication/otp_verification.html', {'email': email})

        try:
            user = CustomUser.objects.get(email=email)  # Use CustomUser
        except CustomUser.DoesNotExist:
            messages.error(request, 'User not found. Please register first.')
            return render(request, 'authentication/otp_verification.html', {'email': email})

        if 'resend_otp' in request.POST:
            resend_otp(user)
            messages.info(request, 'OTP resent successfully.')
            return render(request, 'authentication/otp_verification.html', {'email': email})

        otp_input = ''.join([request.POST.get(f'otp{i}') for i in range(1, 7)])
        if user.otp == otp_input:
            user.otp_verified = True
            user.otp = None
            user.save()
            # Authenticate the user using the username and password
            password = request.session.get('password')  # Retrieve password from session
            user = authenticate(username=user.username, password=password)  # Use CustomUser
            if user is not None:
                auth_login(request, user)
                del request.session['email']  
                del request.session['password']  # Clear password from session
                messages.success(request, 'OTP verified successfully!')
                return redirect('image_generation:dashboard')  
            else:
                messages.error(request, 'Invalid credentials. Please try again.')  
        else:
            messages.error(request, 'Invalid OTP. Please try again.')
    return render(request, 'authentication/otp_verification.html', {'email': email})

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Authenticate the user
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            auth_login(request, user)  # Log the user in
            messages.success(request, 'Login successful!')
            return redirect('image_generation:dashboard')  # Redirect to the dashboard
        else:
            messages.error(request, 'Invalid credentials. Please try again.')
            return render(request, 'authentication/login.html')  # Return to login form

    return render(request, 'authentication/login.html')  # Render the login form for GET requests



def user_logout(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('authentication:signup')  # Redirect to login or home page
    # return redirect('authentication:signup')
