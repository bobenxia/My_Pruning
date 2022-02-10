
def xx(a, b, c):
    print(f"a:{a}, b:{b}, c:{c}")

kwargs = {"b":2, "c":3}
xx(1, **kwargs)