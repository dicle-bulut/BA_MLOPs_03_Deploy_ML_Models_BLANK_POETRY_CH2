from fastapi import FastAPI, Depends, HTTPException, status
 
app = FastAPI()
 
# Simulated function to get the user's role
def get_current_user_role():
    """
    Simulates extracting the user's role.
    In a real application, you'd decode query the a database for authentication (username: password).
    """
    return "admin"  # Change this to "admin" to simulate admin access
    #return "admin"
 
# Dependency to require admin role
def require_admin(user_role: str = Depends(get_current_user_role)):
    if user_role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Admins only."
        )
 
# Public route, accessible by anyone
@app.get("/public")
def public_endpoint():
    return {"message": "This endpoint is open to all users."}
 
# Admin-only route, protected by role dependency
@app.get("/admin", dependencies=[Depends(require_admin)])
def admin_endpoint():
    return {"message": "Welcome, Admin. You have access to this route."}
 
 