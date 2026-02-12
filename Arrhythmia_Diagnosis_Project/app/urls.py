from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('admin-login/', views.admin_login_view, name='admin_login'),
    path('login/', views.user_login_view, name='login'),
    path('register/', views.user_register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),

    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin/records/', views.view_all_ecg_records, name='view_ecg_records'),
    path('training-metrics/', views.view_training_plot, name='training_plot'),

    path('upload/', views.upload_ecg, name='upload_ecg'),
    path('my-records/', views.view_own_ecg_records, name='user_view_ecg_records'),
    path('delete-record/<int:record_id>/', views.delete_ecg_record, name='delete_ecg_record'),

    path('admin/users/', views.admin_user_list, name='admin_user_list'),
    
    path('admin/users/<int:user_id>/edit/', views.admin_user_edit, name='admin_user_edit'),
    path('admin/users/<int:user_id>/delete/', views.admin_user_delete, name='admin_user_delete'),
]
