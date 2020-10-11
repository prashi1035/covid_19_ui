from __future__ import unicode_literals
from django.db import models

class UserManager(models.Manager):
    def validator(self, postData):
        errors = {}
        if (postData['first_name'].isalpha()) == False:
            if len(postData['first_name']) < 2:
                errors['first_name'] = "First name can not be shorter than 2 characters"

        if (postData['last_name'].isalpha()) == False:
            if len(postData['last_name']) < 2:
                errors['last_name'] = "Last name can not be shorter than 2 characters"

        if len(postData['email']) == 0:
            errors['email'] = "You must enter an email"

        if len(postData['password']) < 8:
            errors['password'] = "Password is too short!"
        

        return errors

class User(models.Model):
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add = True)
    updated_at = models.DateTimeField(auto_now = True)
    objects = UserManager()
    
    
    



# <div class="form-group">
#                                 <label>Last Name</label>
#                                 <input name="last_name" type="text" class="form-control">
#                                 {% if messages %} 
#                                 {%for message in messages %} 
#                                 {%if "last_name" in message.tags %}
#                                 <div class="form-group has-danger">
#                                     <div class="form-control-feedback">{{message}}</div>
#                                   </div>
#                                 {%endif%} {%endfor%} {% endif %}
#                             </div>