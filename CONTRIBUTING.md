# Contributing to Volleyball Spike Analyzer

First off, thank you for considering contributing to Volleyball Spike Analyzer! 🏐

## 🎯 How Can I Contribute?

### Reporting Bugs 🐛

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Descriptive title** - Clear and concise
- **Steps to reproduce** - Detailed reproduction steps
- **Expected behavior** - What you expected to happen
- **Actual behavior** - What actually happened
- **Environment details**:
  - OS (Windows/Mac/Linux)
  - Python or R version
  - Package versions
  - Video format and resolution (if applicable)
- **Screenshots/videos** - If applicable
- **Error messages** - Full error output

### Suggesting Enhancements 💡

Enhancement suggestions are welcome! Please include:

- **Clear description** of the enhancement
- **Use cases** - Why this would be useful
- **Possible implementation** - If you have ideas
- **Examples** - Mock-ups or code snippets

### Pull Requests 🔀

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages** (`git commit -m 'Add amazing feature'`)
6. **Push to your branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

## 📝 Development Guidelines

### Python Code Style

- Follow **PEP 8** style guide
- Use **type hints** where appropriate
- Write **docstrings** for all functions (Google style)
- Keep functions **focused and modular**
- Add **comments** for complex logic
- Use **meaningful variable names**

Example:
```python
def calculate_angle(a: Tuple[float, float], 
                    b: Tuple[float, float], 
                    c: Tuple[float, float]) -> float:
    """
    Calculate angle at point b formed by three points.
    
    Args:
        a: First point (x, y) coordinates
        b: Middle point (vertex) (x, y) coordinates
        c: Third point (x, y) coordinates
    
    Returns:
        Angle in degrees (0-180)
    """
    # Implementation...
```

### R Code Style

- Follow **tidyverse style guide**
- Use **roxygen2** documentation
- Prefer **tidyverse functions** over base R
- Use **meaningful variable names** (snake_case)
- Add **comments** for complex operations

Example:
```r
#' Calculate angle at point B
#' 
#' @param a Numeric vector: (x, y) coordinates
#' @param b Numeric vector: (x, y) coordinates  
#' @param c Numeric vector: (x, y) coordinates
#' @return Numeric: Angle in degrees
calculate_angle <- function(a, b, c) {
  # Implementation...
}
```

### Testing

- **Test your changes** before submitting
- Add **test cases** for new features
- Ensure **existing tests pass**
- Test on **different platforms** if possible

### Documentation

- Update **README.md** if adding features
- Update **QUICK_REFERENCE.md** for common tasks
- Add **code comments** for complex logic
- Update **docstrings/roxygen** documentation

## 🌳 Branch Naming

Use descriptive branch names:

- `feature/add-3d-pose` - New features
- `fix/calibration-bug` - Bug fixes
- `docs/update-readme` - Documentation
- `refactor/cleanup-code` - Code improvements
- `test/add-unit-tests` - Testing

## 💬 Commit Messages

Write clear commit messages:

```
Add real-time video processing feature

- Implement frame-by-frame analysis
- Add live visualization option
- Update documentation with examples
- Add tests for new functionality
```

Format:
- **First line**: Brief summary (50 chars or less)
- **Body**: Detailed explanation (wrap at 72 chars)
- Use **present tense** ("Add feature" not "Added feature")
- Reference **issues** if applicable (#123)

## 🎨 Code of Conduct

### Our Standards

- **Be respectful** and inclusive
- **Welcome newcomers** and help them
- **Accept constructive criticism** gracefully
- **Focus on what's best** for the community
- **Show empathy** towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

## 🏗️ Development Setup

### Python Development

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/volleyball-spike-analyzer.git
cd volleyball-spike-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black volleyball_spike_analyzer.py

# Check style
flake8 volleyball_spike_analyzer.py
```

### R Development

```r
# Install dependencies
install.packages(c("tidyverse", "ggplot2", "gridExtra", "signal", "pracma"))

# Install development packages
install.packages(c("testthat", "roxygen2", "devtools", "lintr"))

# Check code style
lintr::lint("volleyball_spike_analyzer.R")

# Run tests
devtools::test()
```

## 🎯 Priority Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- [ ] Real-time video processing optimization
- [ ] Multi-person tracking
- [ ] 3D pose estimation
- [ ] Mobile app (iOS/Android)
- [ ] Web interface
- [ ] More comprehensive tests

### Medium Priority
- [ ] Additional sports support (tennis, basketball, etc.)
- [ ] Advanced ML models for technique classification
- [ ] Performance optimization
- [ ] Better error handling
- [ ] Internationalization (i18n)

### Nice to Have
- [ ] Docker container
- [ ] Cloud deployment guide
- [ ] Video editing integration
- [ ] Comparative analysis dashboard
- [ ] Training recommendation system

## 📚 Resources

- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [R for Data Science](https://r4ds.had.co.nz/)
- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Tidyverse Style Guide](https://style.tidyverse.org/)

## ❓ Questions?

- Open an **issue** for questions
- Check **existing issues** first
- Be patient - we're all volunteers!

## 🙏 Recognition

Contributors will be:
- Listed in **README.md** contributors section
- Mentioned in **release notes**
- Thanked in **commit messages**

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for making Volleyball Spike Analyzer better!** 🎉
