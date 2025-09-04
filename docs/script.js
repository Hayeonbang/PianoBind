// Global functions for tab switching
window.showTab = function(tabNum) {
    // Hide all tabs
    document.getElementById('content1').style.display = 'none';
    document.getElementById('content2').style.display = 'none';
    
    // Remove active class from all tabs
    document.getElementById('tab1').classList.remove('active');
    document.getElementById('tab2').classList.remove('active');
    
    // Show selected tab
    document.getElementById('content' + tabNum).style.display = 'block';
    document.getElementById('tab' + tabNum).classList.add('active');
}

// Button selection
window.selectTag = function(button) {
    // Remove active class from all tag buttons
    var buttons = document.getElementsByClassName('tag-btn');
    for (var i = 0; i < buttons.length; i++) {
        buttons[i].classList.remove('active');
    }
    
    // Add active class to clicked button
    button.classList.add('active');
}