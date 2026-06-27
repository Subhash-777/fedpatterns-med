from fedpatterns.core.app.client_app import client_fn
c1 = client_fn("0")
c2 = client_fn("0")
print(id(c1) == id(c2))
