<?php
/*
Plugin Name: Multimodal Chatbot
Description: Integrates a multimodal chatbot into WordPress
Version: 1.0
Author: Alec Sebastian Moldovan
*/

function chatbot_enqueue_scripts() {
    wp_enqueue_script('chatbot-script', plugin_dir_url(__FILE__) . 'chatbot.js', array('jquery'), '1.0', true);
    wp_enqueue_style('chatbot-style', plugin_dir_url(__FILE__) . 'chatbot.css');

    wp_localize_script('chat-script', 'chatbotConfig', array(
        'backendUrl' => 'http://127.0.0.1:5000' // Local Flask server URL
    ));
}
add_action('wp_enqueue_scripts', 'chatbot_enqueue_scripts');

function chatbot_footer() {
    echo '<div id="chatbot"></div>';
}
add_action('wp_footer', 'chatbot_footer');

// Add an endpoint for the WordPress REST API to fetch content
function get_all_content() {
    $args = array(
        'post_type' => array('post', 'page'),
        'post_status' => 'publish',
        'posts_per_page' => -1,
    );

    $query = new WP_Query($args);
    $content = array();

    if ($query->have_posts()) {
        while ($query->have_posts()) {
            $query->the_post();
            $content[] = array(
                'title' => get_the_title(),
                'content' => get_the_content(),
                'type' => get_post_type(),
            );
        }
    }

    wp_reset_postdata();
    return $content;
}

add_action('rest_api_init', function () {
    register_rest_route('chatbot/v1', '/content', array(
        'methods' => 'GET',
        'callback' => 'get_all_content',
    ));
});
