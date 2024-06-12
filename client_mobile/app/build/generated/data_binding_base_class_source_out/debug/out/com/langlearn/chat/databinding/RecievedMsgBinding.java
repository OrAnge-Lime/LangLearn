// Generated by view binder compiler. Do not edit!
package com.langlearn.chat.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.langlearn.chat.R;
import java.lang.NullPointerException;
import java.lang.Override;
import java.lang.String;

public final class RecievedMsgBinding implements ViewBinding {
  @NonNull
  private final RelativeLayout rootView;

  @NonNull
  public final TextView recieved;

  @NonNull
  public final ImageView sound;

  private RecievedMsgBinding(@NonNull RelativeLayout rootView, @NonNull TextView recieved,
      @NonNull ImageView sound) {
    this.rootView = rootView;
    this.recieved = recieved;
    this.sound = sound;
  }

  @Override
  @NonNull
  public RelativeLayout getRoot() {
    return rootView;
  }

  @NonNull
  public static RecievedMsgBinding inflate(@NonNull LayoutInflater inflater) {
    return inflate(inflater, null, false);
  }

  @NonNull
  public static RecievedMsgBinding inflate(@NonNull LayoutInflater inflater,
      @Nullable ViewGroup parent, boolean attachToParent) {
    View root = inflater.inflate(R.layout.recieved_msg, parent, false);
    if (attachToParent) {
      parent.addView(root);
    }
    return bind(root);
  }

  @NonNull
  public static RecievedMsgBinding bind(@NonNull View rootView) {
    // The body of this method is generated in a way you would not otherwise write.
    // This is done to optimize the compiled bytecode for size and performance.
    int id;
    missingId: {
      id = R.id.recieved;
      TextView recieved = ViewBindings.findChildViewById(rootView, id);
      if (recieved == null) {
        break missingId;
      }

      id = R.id.sound;
      ImageView sound = ViewBindings.findChildViewById(rootView, id);
      if (sound == null) {
        break missingId;
      }

      return new RecievedMsgBinding((RelativeLayout) rootView, recieved, sound);
    }
    String missingId = rootView.getResources().getResourceName(id);
    throw new NullPointerException("Missing required view with ID: ".concat(missingId));
  }
}
