# Developer helper Makefile for sparkx
# Usage: make [target]

.PHONY: help build-ext clean-ext install-dev test wheel clean

help:
	@echo "Available targets:"
	@echo "  build-ext    Install Cython and build C extension in place"
	@echo "  clean-ext    Remove compiled extension and build artifacts"
	@echo "  install-dev  pip install -e . (editable)"
	@echo "  test         Run pytest in quiet mode"
	@echo "  wheel        Build sdist and wheel into dist/"
	@echo "  clean        Remove build/, dist/, egg-info, and __pycache__"

build-ext:
	python -m pip install -U pip setuptools wheel Cython
	python setup.py build_ext --inplace

clean-ext:
	@echo "Cleaning compiled extensions"
	@# Remove compiled shared objects for particle and filter accelerators
	@find src/sparkx -name "_particle_accel.*.so" -delete || true
	@find src/sparkx -name "_particle_accel.*.pyd" -delete || true
	@find src/sparkx -name "_particle_accel.c" -delete || true
	@find src/sparkx -name "_filter_accel.*.so" -delete || true
	@find src/sparkx -name "_filter_accel.*.pyd" -delete || true
	@find src/sparkx -name "_filter_accel.c" -delete || true
	@# Remove any lingering shared objects within src/sparkx (safety net)
	@find src/sparkx -name "*.so" -type f -delete || true
	@rm -rf build

install-dev:
	python -m pip install -e .

test:
	pytest -q

wheel:
	python -m pip install -U build
	python -m build

clean:
	@rm -rf build dist *.egg-info src/sparkx.egg-info
	@# Remove any compiled shared objects and generated C sources in tree
	@find src/sparkx -name "*.so" -type f -delete || true
	@find src/sparkx -name "*.pyd" -type f -delete || true
	@find src/sparkx -name "_particle_accel.c" -type f -delete || true
	@find src/sparkx -name "_filter_accel.c" -type f -delete || true
	@find . -name "__pycache__" -type d -exec rm -rf {} +