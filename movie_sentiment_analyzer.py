import pysrt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from scipy.ndimage import gaussian_filter1d
import torch
import openai
import json
import re
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set tokenizer parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MovieEmotionAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2', openai_api_key=None):
        """Initialize emotion analyzer"""
        # Use a pre-trained sentence transformer model that's specifically designed for semantic tasks
        self.model = SentenceTransformer(model_name)
        self.emotion_dimensions = ['sentiment']  # Remove intensity dimension
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def clean_subtitle_text(self, text):
        """Clean subtitle text by removing HTML tags and extra spaces
        Args:
            text: Raw subtitle text
        Returns:
            cleaned_text: Cleaned text without HTML tags
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing spaces
        text = text.strip()
        return text
    
    def load_subtitle(self, subtitle_path):
        """Load subtitle file"""
        return pysrt.open(subtitle_path)
    
    def get_emotion_scores(self, texts):
        """Get emotion scores for texts using a more sophisticated approach
        Args:
            texts: List of text strings to analyze
        Returns:
            emotion_scores: Array of emotion scores in range [-1, 1]
        """
        # Get text embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        # Define sophisticated sets of positive and negative words
        positive_words = [
            # Joy and Happiness
            'joy', 'happiness', 'delight', 'pleasure', 'ecstasy', 'elation', 'bliss', 'cheer', 'glee',
            # Love and Affection
            'love', 'adore', 'cherish', 'treasure', 'fondness', 'affection', 'devotion', 'passion',
            # Hope and Optimism
            'hope', 'optimism', 'confidence', 'faith', 'trust', 'belief', 'assurance', 'certainty',
            # Gratitude and Appreciation
            'gratitude', 'thankful', 'appreciation', 'blessed', 'fortunate', 'lucky',
            # Success and Achievement
            'success', 'achievement', 'victory', 'triumph', 'accomplishment', 'fulfillment',
            # Peace and Calm
            'peace', 'calm', 'serenity', 'tranquility', 'harmony', 'balance',
            # Excitement and Enthusiasm
            'excitement', 'enthusiasm', 'thrill', 'eager', 'passionate', 'energetic',
            # Pride and Dignity
            'pride', 'dignity', 'honor', 'respect', 'esteem', 'admiration'
        ]
        
        negative_words = [
            # Sadness and Grief
            'sadness', 'grief', 'sorrow', 'melancholy', 'depression', 'despair', 'misery',
            # Anger and Rage
            'anger', 'rage', 'fury', 'wrath', 'outrage', 'resentment', 'bitterness',
            # Fear and Anxiety
            'fear', 'anxiety', 'dread', 'terror', 'panic', 'horror', 'fright',
            # Hate and Disgust
            'hate', 'disgust', 'revulsion', 'contempt', 'loathing', 'abhorrence',
            # Guilt and Shame
            'guilt', 'shame', 'remorse', 'regret', 'repentance', 'contrition',
            # Loneliness and Isolation
            'loneliness', 'isolation', 'alienation', 'abandonment', 'rejection',
            # Failure and Defeat
            'failure', 'defeat', 'loss', 'setback', 'disappointment', 'frustration',
            # Pain and Suffering
            'pain', 'suffering', 'agony', 'torment', 'anguish', 'distress'
        ]
        
        # Get embeddings for positive and negative words
        pos_embeddings = self.model.encode(positive_words, convert_to_tensor=True)
        neg_embeddings = self.model.encode(negative_words, convert_to_tensor=True)
        
        # Calculate average embeddings
        pos_avg = torch.mean(pos_embeddings, dim=0)
        neg_avg = torch.mean(neg_embeddings, dim=0)
        
        # Create sentiment vector (positive - negative)
        sentiment_vector = pos_avg - neg_avg
        
        # Calculate emotion scores
        emotion_scores = torch.matmul(embeddings, sentiment_vector.unsqueeze(1))
        
        # Normalize scores to [-1, 1] range using min-max normalization
        min_score = torch.min(emotion_scores)
        max_score = torch.max(emotion_scores)
        emotion_scores = 2 * (emotion_scores - min_score) / (max_score - min_score) - 1
        
        return emotion_scores.cpu().numpy()
    
    def format_time(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        time = timedelta(seconds=seconds)
        hours = time.seconds // 3600
        minutes = (time.seconds % 3600) // 60
        seconds = time.seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def process_subtitle(self, subtitle_path):
        """Process subtitle file and return timestamps and emotion scores"""
        subs = self.load_subtitle(subtitle_path)
        times = []
        texts = []
        
        for sub in subs:
            start_time = sub.start.ordinal / 1000.0
            # Clean the subtitle text
            text = self.clean_subtitle_text(sub.text)
            
            times.append(start_time)
            texts.append(text)
        
        # Get emotion scores
        emotion_scores = self.get_emotion_scores(texts)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': times,
            'text': texts
        })
        
        # Add emotion score columns
        for i, dim in enumerate(self.emotion_dimensions):
            df[dim] = emotion_scores[:, i]
        
        return df
    
    def analyze_script_structure(self, df):
        """Analyze script structure using OpenAI API
        Args:
            df: DataFrame containing dialogue and timestamps
        Returns:
            act_boundaries: Act boundary timestamps
        """
        # Prepare dialogue text with timestamps
        script_text = "\n".join([f"[{self.format_time(row['time'])}] {row['text']}" 
                               for _, row in df.iterrows()])
        
        # Call OpenAI API to analyze script structure
        prompt = f"""Analyze the following movie dialogue and identify the five-act structure (Act 1: Setup and Inciting Incident; Act 2: Rising Action and Conflict; Act 3: Climax; Act 4: Falling Action and Resolution; Act 5: Denouement).
Please return the result in JSON format, including the timestamp (in HH:MM:SS format) and a very brief description for each act.

Dialogue:
{script_text}

Please return in JSON format as follows (make sure to use proper JSON formatting with commas and quotes):
{{
    "acts": [
        {{"timestamp": "00:00:00", "description": "description"}},
        {{"timestamp": "00:15:00", "description": "description"}},
        {{"timestamp": "00:30:00", "description": "description"}},
        {{"timestamp": "00:45:00", "description": "description"}},
        {{"timestamp": "01:00:00", "description": "description"}}
    ]
}}"""

        try:
            response = openai.ChatCompletion.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": "You are a professional script analyst, skilled in identifying script structure and emotional changes. Always return properly formatted JSON with timestamps in HH:MM:SS format."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Get the response content
            content = response.choices[0].message.content.strip()
            
            # Try to find JSON in the response
            try:
                # First try to parse the entire response as JSON
                result = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON from response: {content}")
                        raise
                else:
                    print(f"No JSON found in response: {content}")
                    raise
            
            # Validate the result structure
            if not isinstance(result, dict) or 'acts' not in result:
                print(f"Invalid result structure: {result}")
                raise ValueError("Invalid result structure")
            
            if not isinstance(result['acts'], list) or len(result['acts']) != 5:
                print(f"Invalid acts structure: {result['acts']}")
                raise ValueError("Invalid acts structure")
            
            # Convert timestamps to seconds
            act_boundaries = []
            for act in result['acts']:
                if not isinstance(act, dict) or 'timestamp' not in act or 'description' not in act:
                    print(f"Invalid act structure: {act}")
                    raise ValueError("Invalid act structure")
                
                # Convert timestamp to seconds
                try:
                    h, m, s = map(int, act['timestamp'].split(':'))
                    start_time = h * 3600 + m * 60 + s
                except ValueError:
                    print(f"Invalid timestamp format: {act['timestamp']}")
                    raise ValueError("Invalid timestamp format")
                
                act_boundaries.append({
                    "start_time": start_time,
                    "description": act['description']
                })
            
            return act_boundaries
            
        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
            print("Falling back to time-based division...")
            # If API call fails, use time-based default division
            total_duration = df['time'].max()
            return [
                {"start_time": 0, "description": "Act 1: Setup and Inciting Incident"},
                {"start_time": total_duration * 0.2, "description": "Act 2: Rising Action and Conflict"},
                {"start_time": total_duration * 0.4, "description": "Act 3: Climax"},
                {"start_time": total_duration * 0.6, "description": "Act 4: Falling Action and Resolution"},
                {"start_time": total_duration * 0.8, "description": "Act 5: Denouement"}
            ]
    
    def get_act_smoothing_factor(self, time, act_boundaries):
        """Get smoothing factor based on act
        Args:
            time: Current timestamp (seconds)
            act_boundaries: Act boundary timestamps
        Returns:
            smoothing_factor: Smoothing factor for the act
        """
        # Determine current act
        current_act = 0
        for i, boundary in enumerate(act_boundaries):
            if time >= boundary["start_time"]:
                current_act = i
        
        # Return smoothing factor based on act type
        smoothing_factors = {
            0: 3.0,  # Act 1: Less smoothing, preserve details for setup
            1: 2.0,  # Act 2: Medium smoothing, show gradual changes
            2: 0.8,  # Act 3: Minimal smoothing for intense climax
            3: 0.1,  # Act 4: Very minimal smoothing for second climax
            4: 2.0   # Act 5: Medium smoothing for resolution
        }
        
        return smoothing_factors[current_act]
    
    def smooth_emotions(self, df):
        """Smooth emotion scores based on script structure"""
        smoothed_df = df.copy()
        
        # Analyze script structure
        act_boundaries = self.analyze_script_structure(df)
        
        # Apply smoothing for each timestamp
        for i in range(len(df)):
            time = df.iloc[i]['time']
            smoothing_factor = self.get_act_smoothing_factor(time, act_boundaries)
            
            # Use Gaussian filtering for smoothing
            window_size = int(smoothing_factor * 10)
            if window_size % 2 == 0:
                window_size += 1
            
            # Get window around current point
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(df), i + window_size // 2 + 1)
            window = df.iloc[start_idx:end_idx]
            
            # Calculate weighted average
            weights = np.exp(-0.5 * ((window['time'] - time) / (smoothing_factor * 10)) ** 2)
            weights = weights / weights.sum()
            
            # Apply smoothing
            for dim in self.emotion_dimensions:
                smoothed_df.iloc[i, smoothed_df.columns.get_loc(dim)] = np.sum(weights * window[dim])
        
        # Adjust intensity based on act
        for i in range(len(smoothed_df)):
            time = smoothed_df.iloc[i]['time']
            # Determine current act
            current_act = 0
            for j, boundary in enumerate(act_boundaries):
                if time >= boundary["start_time"]:
                    current_act = j
            
            # Intensity adjustment factors based on act
            intensity_factors = {
                0: 1.0,  # Act 1: Slight increase for setup
                1: 1.3,  # Act 2: Moderate increase for rising action
                2: 2.0,  # Act 3: Strong increase for climax
                3: 3.0,  # Act 4: Very strong increase for second climax
                4: 1.2   # Act 5: Moderate increase for resolution
            }
            
            # Apply intensity adjustment
            intensity_factor = intensity_factors[current_act]
            smoothed_df.iloc[i, smoothed_df.columns.get_loc('sentiment')] *= intensity_factor
            
            # Ensure we don't exceed the [-1, 1] range
            smoothed_df.iloc[i, smoothed_df.columns.get_loc('sentiment')] = np.clip(
                smoothed_df.iloc[i, smoothed_df.columns.get_loc('sentiment')], -1, 1
            )
        
        return smoothed_df, act_boundaries
    
    def analyze_context(self, df, window_seconds=180):
        """Analyze emotional context
        Args:
            df: Emotion data
            window_seconds: Context window size (seconds)
        """
        # Calculate emotional changes for each point
        context_changes = []
        
        for i in range(len(df)):
            current_time = df.iloc[i]['time']
            
            # Get context window
            context = df[
                (df['time'] >= current_time - window_seconds) & 
                (df['time'] <= current_time + window_seconds)
            ]
            
            if len(context) < 3:  # Ensure enough context
                continue
            
            # Calculate emotional difference with context
            current_sentiment = df.iloc[i]['sentiment']
            context_mean = context['sentiment'].mean()
            context_std = context['sentiment'].std()
            
            # Calculate significance of emotional change
            if context_std > 0:
                z_score = abs(current_sentiment - context_mean) / context_std
            else:
                z_score = 0
            
            # Calculate emotional change persistence
            # Split context into before and after
            mid_time = current_time
            before_context = context[context['time'] < mid_time]
            after_context = context[context['time'] > mid_time]
            
            if len(before_context) > 1 and len(after_context) > 1:
                # Calculate trends
                before_trend = np.polyfit(before_context['time'], before_context['sentiment'], 1)[0]
                after_trend = np.polyfit(after_context['time'], after_context['sentiment'], 1)[0]
                trend_consistency = abs(before_trend - after_trend)
            else:
                trend_consistency = 1
            
            # Calculate emotional change smoothness
            if len(before_context) > 1 and len(after_context) > 1:
                # Calculate emotional change rates
                before_change_rate = np.diff(before_context['sentiment']) / np.diff(before_context['time'])
                after_change_rate = np.diff(after_context['sentiment']) / np.diff(after_context['time'])
                
                # Calculate rate variance, lower variance means smoother change
                before_smoothness = np.var(before_change_rate) if len(before_change_rate) > 0 else 1
                after_smoothness = np.var(after_change_rate) if len(after_change_rate) > 0 else 1
                
                # Combined smoothness
                smoothness = 1 / (1 + before_smoothness + after_smoothness)
            else:
                smoothness = 0
            
            # Combined score: consider significance, persistence, and smoothness
            score = z_score * (1 - trend_consistency) * smoothness
            
            context_changes.append((i, score))
        
        return context_changes
    
    def find_key_moments(self, df, act_boundaries):
        """Find most significant emotional moments for each act
        Args:
            df: Emotion data
            act_boundaries: Act boundary timestamps
        Returns:
            key_moments: List of key moment indices
        """
        key_moments = []
        
        # Find key moments for each act
        for i in range(len(act_boundaries)):
            # Get act start and end times
            start_time = act_boundaries[i]["start_time"]
            end_time = act_boundaries[i+1]["start_time"] if i < len(act_boundaries)-1 else df['time'].max()
            
            # Get dialogue within this act
            act_dialogue = df[(df['time'] >= start_time) & (df['time'] < end_time)]
            
            if len(act_dialogue) == 0:
                continue
            
            if i == 3:  # 4th act
                # For the 4th act, find top 2 maximum and minimum moments
                top_2_max = act_dialogue.nlargest(2, 'sentiment')
                top_2_min = act_dialogue.nsmallest(2, 'sentiment')
                
                # Add all four moments
                key_moments.extend([top_2_max.index[0], top_2_max.index[1], 
                                  top_2_min.index[0], top_2_min.index[1]])
            else:
                # For other acts, find the maximum and minimum moments
                max_sentiment_idx = act_dialogue['sentiment'].idxmax()
                min_sentiment_idx = act_dialogue['sentiment'].idxmin()
                
                # Add both moments
                key_moments.extend([max_sentiment_idx, min_sentiment_idx])
        
        return key_moments
    
    def plot_emotion_curves(self, df, smoothed_df, act_boundaries, title="Emotional Journey"):
        """Plot emotion curves using Plotly with cinematic design"""
        # Create figure
        fig = make_subplots(rows=1, cols=1, vertical_spacing=0.1)
        
        # Convert time to datetime for proper formatting
        smoothed_df['datetime'] = pd.to_datetime(smoothed_df['time'], unit='s')
        
        # Apply additional smoothing for a more natural emotional flow
        smoothed_df['sentiment'] = gaussian_filter1d(smoothed_df['sentiment'], sigma=2.0)
        
        # Define gradient colors with greater contrast
        gradient_colors = [
            '#ffffff',  # Start: pure white
            '#d0e6f6',  # Soft blue
            '#8ab6e1',  # Deeper blue
            '#f7d2dc',  # Soft rose
            '#ebd8f0'   # Light lilac
        ]
        
        # Create continuous gradient background
        total_duration = (smoothed_df['datetime'].max() - smoothed_df['datetime'].min()).total_seconds()
        num_segments = 200  # Increased segments for smoother gradient
        
        for i in range(num_segments):
            t0 = smoothed_df['datetime'].min() + pd.Timedelta(seconds=total_duration * i / num_segments)
            t1 = smoothed_df['datetime'].min() + pd.Timedelta(seconds=total_duration * (i + 1) / num_segments)
            # Blend between adjacent gradient colors
            idx = (i / num_segments) * (len(gradient_colors) - 1)
            i1, i2 = int(idx), min(int(idx) + 1, len(gradient_colors) - 1)
            f = idx - i1
            c1, c2 = gradient_colors[i1], gradient_colors[i2]
            seg_color = self._blend_colors(c1, c2, f)
            fig.add_shape(
                type="rect",
                x0=t0, y0=-1.5, x1=t1, y1=1.5,
                fillcolor=seg_color,
                opacity=0.4,
                layer="below",
                line_width=0,
                row=1, col=1
            )
        
        # Main emotion curve
        fig.add_trace(
            go.Scatter(
                x=smoothed_df['datetime'],
                y=smoothed_df['sentiment'],
                mode='lines',
                name='Emotional Journey',
                line=dict(color='#2c7a7b', width=1.2, shape='spline'),
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Act boundaries and floating labels
        for i, act in enumerate(act_boundaries):
            act_time = pd.to_datetime(act["start_time"], unit='s')
            fig.add_vline(
                x=act_time,
                line_dash="dot",
                line_color="rgba(44,122,123,0.15)",
                line_width=0.6,
                opacity=0.15
            )
            # mid‐point label
            if i < len(act_boundaries) - 1:
                next_time = pd.to_datetime(act_boundaries[i+1]["start_time"], unit='s')
                label_x = act_time + (next_time - act_time) / 2
            else:
                label_x = act_time + (smoothed_df['datetime'].max() - act_time) / 2
            fig.add_annotation(
                x=label_x,
                y=1.4,
                text=f"<b>Act {i+1}</b>",
                showarrow=False,
                font=dict(family='SF Pro Display', size=11, color='rgba(68,68,68,0.7)')
            )
        
        # Key moments (markers + glow)
        key_moments = self.find_key_moments(smoothed_df, act_boundaries)
        act_moments = {}
        for idx in key_moments:
            act_idx = max(i for i, b in enumerate(act_boundaries) if smoothed_df.iloc[idx]['time'] >= b["start_time"])
            act_moments.setdefault(act_idx, []).append(idx)
        
        for moments in act_moments.values():
            for idx in sorted(moments, key=lambda i: smoothed_df.iloc[i]['time']):
                t = smoothed_df.iloc[idx]['datetime']
                s = smoothed_df.iloc[idx]['sentiment']
                text = df.iloc[idx]['text']
                text = text[:50] + "..." if len(text) > 50 else text
                # marker
                fig.add_trace(
                    go.Scatter(
                        x=[t], y=[s],
                        mode='markers',
                        marker=dict(color='#2c7a7b', size=4, line=dict(width=1, color='white')),
                        hoverinfo='text',
                        hovertext=f'"{text}"<br>Time: {self.format_time(smoothed_df.iloc[idx]["time"])}',
                        showlegend=False
                    )
                )
                # glow
                fig.add_trace(
                    go.Scatter(
                        x=[t], y=[s],
                        mode='markers',
                        marker=dict(color='#2c7a7b', size=8, opacity=0.2),
                        hoverinfo='skip',
                        showlegend=False
                    )
                )
        
        # Layout
        fig.update_layout(
            title=dict(
                text=title, x=0.5, y=0.95,
                xanchor='center', yanchor='top',
                font=dict(family='SF Pro Display', size=24, color='#444')
            ),
            height=600, width=1200, showlegend=False,
            yaxis=dict(
                title="Emotional Intensity", range=[-1.5, 1.5],
                showgrid=False, zeroline=False, showline=False,
                title_font=dict(family='SF Pro Display', size=14, color='#444'),
                tickfont=dict(family='SF Pro Display', size=12, color='#444')
            ),
            xaxis=dict(
                title="Time", type='date', tickformat="%H:%M:%S",
                showgrid=False, zeroline=False, showline=True,
                linecolor='rgba(44,122,123,0.15)', linewidth=0.6,
                title_font=dict(family='SF Pro Display', size=14, color='#444'),
                tickfont=dict(family='SF Pro Display', size=12, color='#444')
            ),
            margin=dict(t=100, b=150),  # Increased bottom margin for descriptions
            plot_bgcolor='white', paper_bgcolor='white',
            hovermode='closest',
            hoverlabel=dict(
                bgcolor='rgba(0,0,0,0)',        # fully transparent
                bordercolor='rgba(0,0,0,0)',    # no border
                font=dict(
                    family='SF Pro Display, sans-serif',
                    size=15,
                    color='black'               # pure black text
                )
            )
        )
        
        # Add act descriptions below the plot
        for i, act in enumerate(act_boundaries):
            fig.add_annotation(
                x=0.5,  # Center horizontally
                y=-0.23 - (i * 0.05),  # Position below plot with spacing
                text=f"<b>Act {i+1}</b>: {act['description']}",
                showarrow=False,
                font=dict(family='SF Pro Display', size=15, color='#444'),
                align='center',
                xref='paper',
                yref='paper'
            )
        
        # Custom CSS for cinematic tooltips
        custom_css = """
        <style>
        .hoverlayer .hovertext {
            background: transparent !important;    /* no background */
            border: none !important;               /* no border */
            box-shadow: none !important;           /* no shadow */
            backdrop-filter: none !important;      /* no blur */
            padding: 0 !important;                 /* no padding */
            font-family: 'SF Pro Display', sans-serif !important;
            font-size: 13px !important;
            color: black !important;               /* pure black */
        }
        .hoverlayer .hovertext path,
        .hoverlayer .hovertext polygon {
            fill: transparent !important;
            stroke: none !important;
        }

        /* 新增：确保文字是黑色 */
        .hoverlayer .hovertext tspan {
            fill: rgba(0, 0, 0, 0.6) !important;
            font-style: italic !important;
        }
        </style>
        """
        
        # Export to HTML with CSS injected
        fig.write_html('emotion_curves.html', include_plotlyjs=True, full_html=True)
        with open('emotion_curves.html', 'r') as f:
            html = f.read()
        html = html.replace('</head>', custom_css + '</head>')
        with open('emotion_curves.html', 'w') as f:
            f.write(html)
        
        # Save static image
        fig.write_image('emotion_curves.png')
        
        return fig

    def _blend_colors(self, color1, color2, factor):
        """Blend two colors with a given factor
        Args:
            color1: First color in hex format
            color2: Second color in hex format
            factor: Blending factor (0-1)
        Returns:
            blended_color: Blended color in hex format
        """
        # Convert hex to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert RGB to hex
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        
        # Convert colors to RGB
        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)
        
        # Blend colors
        blended = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * factor for i in range(3))
        
        return rgb_to_hex(blended)

def main():
    # Set OpenAI API key
    openai_api_key = input("Enter OpenAI API key (press Enter to use default analysis): ")
    
    analyzer = MovieEmotionAnalyzer(openai_api_key=openai_api_key if openai_api_key else None)
    
    # Process subtitle file
    subtitle_path = input("Enter subtitle file path (.srt format): ")
    
    try:
        # Process subtitle
        df = analyzer.process_subtitle(subtitle_path)
        
        # Apply smoothing
        smoothed_df, act_boundaries = analyzer.smooth_emotions(df)
        
        # Plot emotion curves
        fig = analyzer.plot_emotion_curves(df, smoothed_df, act_boundaries)
        
        print("Emotion analysis complete!")
        print("- Emotion curves saved as 'emotion_curves.html' and 'emotion_curves.png'")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main() 